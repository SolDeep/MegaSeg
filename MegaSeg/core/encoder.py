 
import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.autograd import Function
import os
import copy
import ctypes
import collections.abc as container_abcs
from itertools import repeat
from dataclasses import dataclass
from typing import Any,NamedTuple, Optional, Sequence, Tuple, List
from .loader_modules import BaseLoaderModule
import torchvision.transforms as transforms
from torch.cuda.amp import custom_bwd, custom_fwd
 

def _ntuple(n):
    def parse(x, default=0):
        if isinstance(x, container_abcs.Iterable):
            if len(x) == n: 
                return x
            elif len(x) == n-1: 
                return tuple([default, *x])
            else: 
                return tuple(repeat(x[0], n))
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)

@dataclass
class _TileEncoderArguments: 
    loader_module: BaseLoaderModule
    conv_module: nn.Module 
    tile_size: int
    emb_crop_size: int
    emb_stride_size: int
    skip_no_grad: bool
    cache_background_forward: bool
    cache_background_backward: bool 
    mask_batch: Optional[Tensor] 
    skip_layers: list 
    enable_grad: bool 
    hook_side: ctypes.pointer 

class Sides(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int



class TileEncoder(nn.Module): 
 

    def __init__(
        self,
        loader_module: nn.Module,
        conv_module: nn.Module,  
        tile_size: int = 4096,
        emb_crop_size: int = 7,
        emb_stride_size: int = 32,
        finetune_layers:int=4,
        skip_no_grad: bool = True,
        cache_background_forward: bool = True,
        cache_background_backward: bool = True, 
        skip_layers  = [[0,3],[3,5],5,6,7],
        hook_side: ctypes.pointer= None 
       
    ):
        super().__init__()

        if not isinstance(loader_module, BaseLoaderModule):
            raise ValueError("loader_module should be an instance of BaseLoaderModule.")
        
        if not isinstance(skip_layers, List):
            raise ValueError("skip_layers should be a list .")

        self.loader_module = loader_module
        self.conv_module = conv_module  
        self.tile_size = tile_size
        self.emb_crop_size = emb_crop_size
        self.emb_stride_size = emb_stride_size
        self.skip_no_grad = skip_no_grad
        self.cache_background_forward = cache_background_forward
        self.cache_background_backward = cache_background_backward,
        self.skip_layers = skip_layers 
        self.hook_side = hook_side 
        self.finetune_layers = finetune_layers 

    def forward(self, image_batch: torch.Tensor, mask_batch: torch.Tensor) -> torch.Tensor:
        
        self.loader_module.randomize()  
        params = self.conv_module.parameters() if self.finetune_layers >= 4 else  self.conv_module[-self.finetune_layers:].parameters()
      
        output = _TileEncoderConvolutional.apply( 
                image_batch, 
                _TileEncoderArguments(
                    loader_module=self.loader_module,
                    conv_module=self.conv_module,  
                    tile_size=self.tile_size,
                    emb_crop_size=self.emb_crop_size,
                    emb_stride_size=self.emb_stride_size,
                    skip_no_grad=self.skip_no_grad,
                    cache_background_forward=self.cache_background_forward,
                    cache_background_backward=self.cache_background_backward, # type: ignore
                    mask_batch = mask_batch,
                    skip_layers = self.skip_layers, # type: ignore
                    enable_grad=False,
                    hook_side = self.hook_side  
                ),
                *params 
                )   

        return output 



class _TileEncoderConvolutional(Function):
     
    @staticmethod
    @custom_fwd
    def forward(
        ctx, 
        image_batch, 
        arguments,
        *conv_parameters,
    ):
       
        #find a way to make image and mask to both be NHWC or NCHW format
        # Save parameters  
        ctx.image_batch = image_batch 
        ctx.arguments = arguments
        ctx.conv_parameters = conv_parameters 

        mask_batch = arguments.mask_batch
        if mask_batch is not None:
            if len(mask_batch.shape)   > 3:
                    mask_batch = mask_batch.permute(0, 3, 1, 2) 
       
        # Load arguments
        loader_module = arguments.loader_module
        
     
        loader_module.training = arguments.conv_module.training
        # Create a background tile cache if required
         

        # Calculate the tile number
        tile_dimensions = _TileEncoderConvolutional._compute_tile_dimensions(
            image_batch,
            arguments,
        ) 
      
        # Hint loader module the future accesses.
        _TileEncoderConvolutional._hint_loader_module(
            image_batch,
            mask_batch,
            tile_dimensions,
            arguments,
        ) 
         
        # Forward convolutional  
        with torch.set_grad_enabled(arguments.enable_grad):  # Do no store any feature maps
            # Iterate tiles
            emb_tiles = [] 
            for tile_y in range(tile_dimensions[1]):
                emb_tiles_row = [] 
                for tile_x in range(tile_dimensions[0]):
                    # Load image tile
                    ( tile_coord, tile_size,  ) = _TileEncoderConvolutional._compute_image_tile_coord( image_batch,  arguments, (tile_x, tile_y),  )
                     
                    image_tile_batch, mask_tile_batch = loader_module(image_batch, mask_batch,  tile_coord,  tile_size, )  
                                
                    # Do forward
                    emb_tile, _ = _TileEncoderConvolutional._encoder_forward_tile(
                        image_tile_batch,  arguments,  (tile_x, tile_y),
                        tile_dimensions,  
                        require_skip=False   )  
                 
                    del image_tile_batch, mask_tile_batch
                    emb_tiles_row.append(emb_tile) 
                emb_tiles.append(emb_tiles_row) 

            # Compute the look-up table for the coordinates of embedding tiles
            emb_tile_coord_lut = _TileEncoderConvolutional._compute_emb_tile_coord_lut( emb_tiles)
            
            # Concatenate tiles to get the embedding feature map
            emb_rows = [torch.cat(emb_tiles_row, dim=3) for emb_tiles_row in emb_tiles]
            emb = torch.cat(emb_rows, dim=2) 
            #memory management
            # del skip 
            del emb_tiles 
            del emb_rows
            del emb_tiles_row
            # del image_batch 
 
        ctx.emb_tile_coord_lut = emb_tile_coord_lut   
     
        return emb,mask_batch 

    @staticmethod
    @custom_bwd
    def backward(
        ctx: Any,
        grad_emb,
        _, 
    ) -> Sequence[Optional[torch.Tensor]]:
        """ 

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
        # image_batch = image_batch.float() 
        mask_batch = None
        arguments = ctx.arguments
        conv_parameters = ctx.conv_parameters  
        emb_tile_coord_lut = ctx.emb_tile_coord_lut   

        # Load arguments
        loader_module = arguments.loader_module  
        # Calculate the tile number
        tile_dimensions = _TileEncoderConvolutional._compute_tile_dimensions(
            image_batch,
            arguments,
        ) 
        # Hint loader module the future accesses. 
        _TileEncoderConvolutional._hint_loader_module(
            image_batch,
            mask_batch,
            tile_dimensions,
            arguments,
        )
        # Iterate tiles 
        grad_conv_parameters = [
            torch.zeros_like(parameter, device=parameter.device)
            for parameter in conv_parameters
        ]
         
        for tile_y in range(tile_dimensions[1]):
            for tile_x in range(tile_dimensions[0]):
                arguments.hook_side.contents.x_min= tile_x
                arguments.hook_side.contents.y_min= tile_y
               
                with torch.enable_grad():
                    # Get the gradients w.r.t. the embedding tile
                    ( grad_emb_tile_coord, 
                     grad_emb_tile_size
                    ) = _TileEncoderConvolutional._use_emb_tile_coord_lut(
                        emb_tile_coord_lut,
                        (tile_x, tile_y)
                        )
                    grad_emb_tile = grad_emb[:,:,
                        grad_emb_tile_coord[1] : grad_emb_tile_coord[1]
                        + grad_emb_tile_size[1],
                        grad_emb_tile_coord[0] : grad_emb_tile_coord[0]
                        + grad_emb_tile_size[0],
                    ] 
                  
                    if ( arguments.skip_no_grad  and torch.count_nonzero(grad_emb_tile).item() == 0  ):
                        _TileEncoderConvolutional._prefetch_next(arguments)
                        continue

                    # Load image tile
                    (tile_coord, tile_size, ) = _TileEncoderConvolutional._compute_image_tile_coord( image_batch, arguments, (tile_x, tile_y))
                 
                    image_tile_batch,_ = loader_module( image_batch, mask_batch, tile_coord, tile_size, ) 

                    # Re-compute forward convolutional.  
                    emb_tile, _ = _TileEncoderConvolutional._encoder_forward_tile(
                        image_tile_batch,
                        arguments,
                        (tile_x, tile_y),
                        tile_dimensions, 
                        require_skip=False )
                     
                    #compute the decoder path
                    # Compute the partial gradients w.r.t. the parameters in the
                    # convolutional module 
                    partial_grad_conv_parameters = torch.autograd.grad(
                        [emb_tile],
                        conv_parameters,
                        [grad_emb_tile], 
                    )     

                with torch.no_grad():
                    # Accumulate partial gradients
                    for idx, partial_grad_conv_parameter in enumerate(
                        partial_grad_conv_parameters
                    ):
                         
                        grad_conv_parameters[idx] += partial_grad_conv_parameter
                del image_tile_batch
        del _
        # del __
        del mask_batch 
        del image_batch 
        del grad_emb 
        return (None, None) + tuple(grad_conv_parameters)

 


    @staticmethod
    def _encoder_forward_tile(
        image_tile_batch: torch.Tensor,
        arguments: _TileEncoderArguments,
        tile_indices: Tuple[int, int],
        tile_dimensions: Tuple[int, int], 
        require_skip: Optional[bool] = False
    ) -> torch.Tensor:
        # Get arguments
        conv_module = arguments.conv_module 
        emb_crop_size = arguments.emb_crop_size
        skip_layers = arguments.skip_layers
 
        # None.
        emb_tile = None 
      
        # Do convolutions when cache miss
        skip = []
        features = image_tile_batch 
        #Skip Connection Required size  
        for inter_m,layer in enumerate(skip_layers[:-1]):
            features = conv_module[layer[0]:layer[1]](features) \
            if isinstance(layer, list) else conv_module[layer](features)  
            if require_skip: 
                    skip.append(features.detach().to('cpu'))   
        emb_tile = conv_module[skip_layers[-1]](features) 
              
        #crop out invalide borders
        emb_tile = _TileEncoderConvolutional._crop_invalid_borders(emb_tile,
                                                               emb_crop_size,
                                                               tile_indices,
                                                               tile_dimensions)
      
    
        del features
        return emb_tile, skip
    

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
        arguments: _TileEncoderArguments,
    ) -> Tuple[int, int]:
        # Get arguments
        tile_size = arguments.tile_size
        emb_crop_size = arguments.emb_crop_size
        emb_stride_size = arguments.emb_stride_size
        _, _, height, width,  = image_batch.shape  #image is still in shape BHWC 

        # Compute tile dimensions 
        overlapping_size = emb_crop_size * emb_stride_size  * 2 
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
        mask_batch: torch.Tensor,
        tile_dimensions: Tuple[int, int],
        arguments: _TileEncoderArguments,
    ) -> None:
        # Get arguments
        loader_module = arguments.loader_module

        # Calculate tile coordinates and sizes that will be accessed, and hint the
        # loader.
        tile_coords = []
        tile_sizes = []
        for tile_y in range(tile_dimensions[1]):
            for tile_x in range(tile_dimensions[0]):
                tile_coord, tile_size = _TileEncoderConvolutional._compute_image_tile_coord(
                    image_batch,
                    arguments,
                    (tile_x, tile_y),
                )
                tile_coords.append(tile_coord)
                tile_sizes.append(tile_size)

        loader_module.hint_future_accesses(image_batch,mask_batch, tile_coords, tile_sizes)

    @staticmethod
    def _prefetch_next(arguments: _TileEncoderArguments) -> None:
        loader_module = arguments.loader_module
        loader_module.prefetch_next()

    @staticmethod
    def _compute_image_tile_coord(
        image_batch: torch.Tensor,
        arguments: _TileEncoderArguments,
        tile_indices: Tuple[int, int],
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # Get arguments
        tile_size = arguments.tile_size
        emb_crop_size = arguments.emb_crop_size
        emb_stride_size = arguments.emb_stride_size

        # Compute coord and size
        _, _, height, width = image_batch.shape
        overlapping_size = emb_crop_size * emb_stride_size  * 2
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


class _BackgroundTileCache:
    def __init__(self):
        self.cache = []

    def __getitem__(
        self,
        tile: torch.Tensor,
    ) -> Any:
        # Get basic info of the tile.
        shape = tuple(tile.shape)
        pixel_values = tuple(tile[0, :, 0, 0].cpu().numpy())

        # Look for the cache entry.
        result = None
        for item in self.cache:
            if item["shape"] == shape and item["pixel_values"] == pixel_values:
                result = item["result"]

        # If cache miss, just return.
        if result is None:
            return None

        # Check if the tile is background tile. If not, return.
        if (
            torch.count_nonzero(
                tile - tile[0, :, 0, 0][np.newaxis, :, np.newaxis, np.newaxis]
            )
            != 0
        ):
            return None

        return result

    def __setitem__(
        self,
        tile: torch.Tensor,
        result: Any,
    ) -> None:
        # Get basic info of the tile.
        shape = tuple(tile.shape)
        pixel_values = tuple(tile[0, :, 0, 0].cpu().numpy())

        # Check if the tile is background tile. If not, return.
        if (
            torch.count_nonzero(
                tile - tile[0, :, 0, 0][np.newaxis, :, np.newaxis, np.newaxis]
            )
            != 0
        ):
            return

        # Raise error if there is the same entry.
        for item in self.cache:
            if item["shape"] == shape and item["pixel_values"] == pixel_values:
                raise ValueError(
                    "The _BackgroundTileCache already stores the same entry."
                )

        # Append the cache entry
        self.cache.append(
            {
                "pixel_values": pixel_values,
                "shape": shape,
                "result": result,
            }
        )
    def __len__(self):
        return len(self.cache)
