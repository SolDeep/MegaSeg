from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor # type: ignore
from torch import nn # type: ignore 
import ctypes
from torch.cuda.amp import custom_bwd, custom_fwd
 
 
@dataclass
class Sides():
    x_min: int
    y_min: int
    x_max: int
    y_max: int  

@dataclass
class _TileDecoderArguments:
    """
    Arguments for `_TileDecoderConvolutional`.

    Args:
        See descriptions in `Hms2Model`.
    """ 
    image_batch: torch.Tensor
    # ski_coonection: np.ndarray
    encoder_model: nn.Module 
    No_stream: nn.Module 
    conv_module: nn.Module 
    loader_module: nn.Module 
    emb_crop_size: int
    skip_layers: list
    tile_size: int
    encoder_tile_size: int
    emb_stride_size: int
    encoder_crop_size: int
    skip_no_grad: bool
    cache_background_forward: bool
    cache_background_backward: bool  
    gather_grad: ctypes.pointer  
    hook_side: ctypes.pointer  

class TileDecoder(nn.Module):
    """
    A torch module implementing TileDecoder.

    """

    def __init__(
        self, 
        loader_module: nn.Module, 
        encoder: nn.Module, 
        No_stream: nn.Module, 
        decoder: nn.Module, 
        tile_size: int = 64,
        encoder_tile_size: int =64,
        emb_crop_size: int = 7,
        emb_stride_size: int = 32,
        skip_layers: list =[0,1],
        skip_no_grad: bool = True,
        encoder_crop_size: int = 7,
        cache_background_forward: bool = False,
        cache_background_backward: bool = False, 
        gather_grad: ctypes.pointer = None,
        hook_side: ctypes.pointer = None
    ):
        super().__init__()
        #NOTE: 
       
        self.loader_module = loader_module
        self.encoder_model = encoder  
        self.No_stream = No_stream  
        self.conv_module = decoder  
        self.tile_size = tile_size
        self.encoder_tile_size = encoder_tile_size
        self.emb_crop_size = emb_crop_size
        self.emb_stride_size = emb_stride_size
        self.encoder_crop_size = encoder_crop_size
        self.skip_no_grad = skip_no_grad
        self.skip_layers = skip_layers
        self.cache_background_forward = cache_background_forward
        self.cache_background_backward = cache_background_backward 
        self.gather_grad = gather_grad 
        self.hook_side = hook_side 

    def forward(self, image_batch, 
                encoder_output_batch): # type: ignore
        """
        Implement how tensors flow in a HMS2 model.

        Args:
            image_batch (torch.Tensor): An image batch in NHWC and uint8 dtype.
            encoder_output_batch (torch.Tensor): encoder output NHWC.
            mask_batch (torch.Tensor): An image mask batch in NHWC and uint8 dtype.

        Returns:
            output (torch.Tensor): The output of dense_module.
        """  
        output = _TileDecoderConvolutional.apply(
            encoder_output_batch, 
            _TileDecoderArguments(
                loader_module = self.loader_module,   
                image_batch=image_batch,
                conv_module=self.conv_module,  
                encoder_model=self.encoder_model,  
                No_stream=self.No_stream,  
                tile_size=self.tile_size,
                encoder_tile_size=self.encoder_tile_size,
                emb_crop_size=self.emb_crop_size,
                emb_stride_size=self.emb_stride_size,
                encoder_crop_size=self.encoder_crop_size,
                skip_no_grad=self.skip_no_grad,
                skip_layers=self.skip_layers,
                cache_background_forward=self.cache_background_forward,
                cache_background_backward=self.cache_background_backward,  # type: ignore 
                gather_grad=self.gather_grad,
                hook_side=self.hook_side 
            ),
            *self.conv_module.parameters()
        ) 
        return output 

class _TileDecoderConvolutional(torch.autograd.Function): # type: ignore
    """
    The core part of HMS2 that implements tiling in the convolutional part and backward
    re-computations. Using torch.autograd.Function instead of torch.nn.Module is
    because only torch.autograd.Function can rewrite custom backward operations.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx: Any,
        image_batch: torch.Tensor, # type: ignore
        arguments: _TileDecoderArguments,
        *conv_parameters: torch.Tensor,# type: ignore
    ) -> Tuple[Tensor,...]:
        """
        HMS2 forward-convolutional.

        Args:
            ctx (Any):
                See PyTorch documentations.
            image_batch (torch.Tensor): An image batch in NHWC and uint8 dtype.
            mask_batch (torch.Tensor): An image mask batch in NCHW and uint8 dtype.
            arguments (_TileDecoderArguments):
                See descriptions in `_TileDecoderArguments`.
            conv_parameters (list of torch.Tensor):
                A list retrieved by calling `conv_module.parameters()`.

        Returns:
            emb (torch.Tensor): The resulting embedding feature map.
        """

        
        with torch.enable_grad():
            image_batch = arguments.No_stream(image_batch)
        # Save parameters 
        ctx.image_batch = image_batch
        ctx.arguments = arguments
        ctx.conv_parameters = conv_parameters  
        loader_module = arguments.loader_module 
  
        tile_dimensions = _TileDecoderConvolutional._compute_tile_dimensions(
            image_batch,
            arguments,
        )
        
        # Forward convolutional 
        with torch.no_grad(): # type: ignore # Do no store any feature maps
            # Iterate tiles
            emb_tiles = [] 
            for tile_y in range(tile_dimensions[1]):
                emb_tiles_row = []  
                for tile_x in range(tile_dimensions[0]):
                    #Recompute the encoder forward 
                    ( tile_coord, tile_size,  ) = _TileDecoderConvolutional._compute_image_tile_coord( 
                        arguments.image_batch,  arguments, (tile_x, tile_y), True )
                    
                    image_tile_batch_forward, _  = loader_module(arguments.image_batch,  
                                                        None,  
                                                        tile_coord,  tile_size, ) 
                    
                
                    skip_connection = _TileDecoderConvolutional._encoder_forward_tile(
                            image_tile_batch_forward,
                            arguments,  
                            
                        )
                     
                     #decoder start here
                    (tile_coord, tile_size ) = _TileDecoderConvolutional._compute_image_tile_coord( 
                        image_batch,  
                        arguments, 
                        (tile_x, tile_y),  
                    )
                    
                    emb_tile = image_batch[:,:,
                        tile_coord[1] : tile_coord[1]
                        + tile_size[1],
                        tile_coord[0] : tile_coord[0]
                        + tile_size[0],
                    ] 
               
                    emb_tile = _TileDecoderConvolutional._decoder_forward_tile(emb_tile.detach(),
                                                                            arguments,
                                                                            (tile_x, tile_y), 
                                                                            tile_dimensions,
                                                                            skip_connection
                                                                            ) 
                    emb_tiles_row.append(emb_tile) 
                emb_tiles.append(emb_tiles_row) 

            # Compute the look-up table for the coordinates of embedding tiles
            emb_tile_coord_lut = _TileDecoderConvolutional._compute_emb_tile_coord_lut(
                emb_tiles
            )
     
            # Concatenate tiles to get the embedding feature map
            emb_rows = [torch.cat(emb_tiles_row, dim=3) for emb_tiles_row in emb_tiles] # type: ignore
            emb = torch.cat(emb_rows, dim=2)  # type: ignore
 

        #memory management
        del emb_tiles_row 
        del emb_tile 
        del emb_rows 
        del skip_connection
      
        # Save the look-up table
        ctx.emb_tile_coord_lut = emb_tile_coord_lut 
       
       
        return emb

    @staticmethod
    @custom_bwd
    def backward(
        ctx: Any,
        grad_emb: torch.Tensor, # type: ignore
    ) -> Sequence[Optional[torch.Tensor]]: # type: ignore
        """
        HMS2 backward-convolutional.

        Args:
            ctx (Any):
                See PyTorch documentations.
            grad_emb (torch.Tensor): The gradients w.r.t. the embedding feature map.

        Returns:
            grad_image_batch (NoneType): Remain None.
            grad_arguments (NoneType): Remain None.
            grad_conv_parameters (tuple):
                A tuple of the gradients w.r.t. parameters in the convolutional module.
        """
        
        # Load saved parameters
        image_batch = ctx.image_batch 
        arguments = ctx.arguments
        conv_parameters = ctx.conv_parameters 
        emb_tile_coord_lut = ctx.emb_tile_coord_lut
        loader_module  = arguments.loader_module

        tile_dimensions = _TileDecoderConvolutional._compute_tile_dimensions(
            image_batch,
            arguments,
        )
 
        # Iterate tiles
        grad_conv_parameters = [
            torch.zeros_like(parameter, device=parameter.device) # type: ignore
            for parameter in ctx.conv_parameters
        ]
        intput_grad_grad = torch.zeros_like(image_batch, device=image_batch.device)
        arguments.hook_side.contents.x_max= tile_dimensions[0]
        arguments.hook_side.contents.y_max= tile_dimensions[1]
    
        skip_connection = 0
        for tile_y in range(tile_dimensions[1]): 
            for tile_x in range(tile_dimensions[0]):
                arguments.hook_side.contents.x_min= tile_x
                arguments.hook_side.contents.y_min= tile_y  
                    
                with torch.enable_grad(): # type: ignore 
                    # Get the gradients w.r.t. the embedding tile
                    (grad_emb_tile_coord, grad_emb_tile_size,) = _TileDecoderConvolutional._use_emb_tile_coord_lut(emb_tile_coord_lut, (tile_x, tile_y),)
                    
                    grad_emb_tile = grad_emb[:,:,
                        grad_emb_tile_coord[1] : grad_emb_tile_coord[1]
                        + grad_emb_tile_size[1],
                        grad_emb_tile_coord[0] : grad_emb_tile_coord[0]
                        + grad_emb_tile_size[0],
                    ] 
                   
                    # Skip computing this tile if all the gradients are 0
                    if ( arguments.skip_no_grad and 
                        torch.count_nonzero(grad_emb_tile).item() == 0 ): # type: ignore 
                        continue
                    
                    with torch.no_grad(): 
                        #Recompute the encoder forward 
                        ( tile_coord, tile_size,  ) = _TileDecoderConvolutional._compute_image_tile_coord( 
                            arguments.image_batch,  arguments, (tile_x, tile_y), True  )
                        
                        image_tile_batch_forward,_  = loader_module(arguments.image_batch,  
                                                            None,  
                                                            tile_coord,  tile_size, )  
                        skip_connection = _TileDecoderConvolutional._encoder_forward_tile(
                                image_tile_batch_forward,
                                arguments,  
                            )
                    
                    for i in range(len(skip_connection)):
                        skip_connection[i].requires_grad_(True).retain_grad()

                    # Load image tile
                    (tile_coord, tile_size, ) = _TileDecoderConvolutional._compute_image_tile_coord(
                        image_batch, arguments, (tile_x, tile_y))
                    
                    
                    emb_tile_ =  image_batch[:, :,
                                    tile_coord[1] : tile_coord[1] + tile_size[1],
                                    tile_coord[0] : tile_coord[0] + tile_size[0]
                                    ]  
               
                    emb_tile_in = emb_tile_.detach().requires_grad_(True) 
                  
                    emb_tile = _TileDecoderConvolutional._decoder_forward_tile(emb_tile_in,
                                                                    arguments,
                                                                    (tile_x, tile_y), 
                                                                    tile_dimensions,
                                                                    skip_connection
                                                                    )
 
                    # Compute the partial gradients w.r.t. the parameters   
                    partial_grad_conv_parameters = torch.autograd.grad(
                        [emb_tile],
                        conv_parameters,
                        [grad_emb_tile], 
                        retain_graph=True 
                    ) 

                    with torch.no_grad(): 
                        arguments.gather_grad.contents.value= True 

                    # Compute the partial gradients w.r.t. the input 
                    grad_input = torch.autograd.grad(emb_tile, emb_tile_in, grad_outputs=grad_emb_tile) 
 
                with torch.no_grad(): 
                    intput_grad_grad[:, :,
                                tile_coord[1] : tile_coord[1] + tile_size[1],
                                tile_coord[0] : tile_coord[0] + tile_size[0]] += grad_input[0]
                
                    del emb_tile_in
                    del emb_tile
                    del grad_input
                    del skip_connection
                    del grad_emb_tile
              
                    for idx, partial_grad_conv_parameter in enumerate(
                        partial_grad_conv_parameters
                    ):
                        
                        grad_conv_parameters[idx] += partial_grad_conv_parameter 
                    del partial_grad_conv_parameters 
                    arguments.gather_grad.contents.value= False 
 
        image_batch.backward(intput_grad_grad)
        del intput_grad_grad
        del image_batch
        
        return (None, None) + tuple(grad_conv_parameters)

    @staticmethod
    def _decoder_forward_tile(
        image_tile_batch: torch.Tensor,
        arguments: _TileDecoderArguments,
        tile_indices: Tuple[int, int], 
        tile_dimensions: Tuple[int, int],
        skip:Tuple[Tensor, ...]
        )-> torch.Tensor:
        conv_decoder = arguments.conv_module 
        emb_crop_size = arguments.emb_crop_size   
        
        emb_tile = conv_decoder(image_tile_batch,skip) 
        emb_tile = _TileDecoderConvolutional._crop_invalid_borders(emb_tile,
                                                            emb_crop_size,
                                                            tile_indices,
                                                            tile_dimensions)
 
        return emb_tile  
    @staticmethod
    def _encoder_forward_tile(
        image_tile_batch: torch.Tensor,
        arguments: _TileDecoderArguments,   
    ) -> torch.Tensor:
        # Get arguments
        encoder = arguments.encoder_model  
        skip_layers = arguments.skip_layers  
      
        skip = []
        features = image_tile_batch  
        for layer in skip_layers[:-1]:
            features = encoder[layer[0]:layer[1]](features) \
            if isinstance(layer, list) else encoder[layer](features) 
            skip.append(features.detach())
        del features
        return skip  

    @staticmethod
    def _crop_invalid_borders(
        emb_tile: torch.Tensor,
        emb_crop_size: int,
        tile_indices: Tuple[int, int],
        tile_dimensions: Tuple[int, int], 
    ):
        tile_x, tile_y = tile_indices
        _, _, emb_tile_height, emb_tile_width = emb_tile.shape
        left = emb_crop_size if tile_x != 0 else 0
        right = -emb_crop_size if tile_x != tile_dimensions[0] - 1 else emb_tile_width
        top = emb_crop_size if tile_y != 0 else 0
        bottom = -emb_crop_size if tile_y != tile_dimensions[1] - 1 else emb_tile_height 
        return  emb_tile[
                :, :,
                top:bottom,
                left:right,
            ]
    
    @staticmethod
    def _compute_tile_dimensions(
        image_batch: torch.Tensor,
        arguments: _TileDecoderArguments,
    ) -> Tuple[int, int]:
        # Get arguments
        tile_size = arguments.tile_size
        encoder_crop_size = arguments.encoder_crop_size
        emb_stride_size = arguments.emb_stride_size
        _, _ , height, width = image_batch.shape  #NCHW
        # Compute tile dimensions
    
        overlapping_size = (encoder_crop_size * emb_stride_size * 2 ) // emb_stride_size
        tile_width = (
            max(0, int(np.ceil((width - tile_size) / (tile_size - overlapping_size))))
            + 1
        )
        tile_height = (
            max(0, int(np.ceil((height - tile_size) / (tile_size - overlapping_size))))
            + 1
        )
        return (tile_width, tile_height)

    @staticmethod
    def _hint_loader_module(
        image_batch: torch.Tensor,
        # mask_batch: torch.Tensor,
        tile_dimensions: Tuple[int, int],
        arguments: _TileDecoderArguments,
    ) -> None:
        # Get arguments
        loader_module = arguments.loader_module

        # Calculate tile coordinates and sizes that will be accessed, and hint the
        # loader.
        tile_coords = []
        tile_sizes = []
        for tile_y in range(tile_dimensions[1]):
            for tile_x in range(tile_dimensions[0]):
                tile_coord, tile_size = _TileDecoderConvolutional._compute_image_tile_coord(
                    image_batch,
                    arguments,
                    (tile_x, tile_y),
                )
                tile_coords.append(tile_coord)
                tile_sizes.append(tile_size)

        loader_module.hint_future_accesses(image_batch, tile_coords, tile_sizes)

    @staticmethod
    def _prefetch_next(arguments: _TileDecoderArguments) -> None:
        loader_module = arguments.loader_module
        loader_module.prefetch_next()

    @staticmethod
    def _compute_image_tile_coord(
        image_batch: torch.Tensor,
        arguments: _TileDecoderArguments,
        tile_indices: Tuple[int, int],
        is_encoder:bool= False
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # Get arguments
        if is_encoder:
            tile_size = arguments.encoder_tile_size
        else:
            tile_size = arguments.tile_size
        encoder_crop_size = arguments.encoder_crop_size
        emb_stride_size = arguments.emb_stride_size

        # Compute coord and size
        _, _ , height, width, = image_batch.shape #NCHW
        if is_encoder:
            overlapping_size = (encoder_crop_size * emb_stride_size * 2)
        else:    
            overlapping_size = (encoder_crop_size * emb_stride_size * 2) //  emb_stride_size 
        tile_x, tile_y = tile_indices
        coord_x = tile_x * (tile_size - overlapping_size)
        coord_y = tile_y * (tile_size - overlapping_size)
        size_x = min(tile_size, width - coord_x)
        size_y = min(tile_size, height - coord_y)

        return (coord_x, coord_y), (size_x, size_y)

    @staticmethod
    def _compute_emb_tile_coord_lut(
        emb_tiles: Sequence[Sequence[torch.Tensor]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        widths = np.array([emb_tile.shape[3] for emb_tile in emb_tiles[0]])
        cum_widths = np.cumsum(widths)

        heights = np.array([row_emb_tiles[0].shape[2] for row_emb_tiles in emb_tiles])
        cum_heights = np.cumsum(heights)

        return widths, cum_widths, heights, cum_heights

    @staticmethod
    def _use_emb_tile_coord_lut(
        lut: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tile_indices: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        widths, cum_widths, heights, cum_heights = lut
        tile_x, tile_y = tile_indices

        coord_x = cum_widths[tile_x] - widths[tile_x]
        coord_y = cum_heights[tile_y] - heights[tile_y]
        size_x = widths[tile_x]
        size_y = heights[tile_y]

        return (coord_x, coord_y), (size_x, size_y)


 