"""
3D U-Net Implementation - Configurable and Generalizable

Supports variable depth, channels, activations, and other parameters
as specified in UNet_pure.py configuration.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ConvBlock3D(nn.Module):
    """3D convolution block with configurable parameters."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 padding: int = 1, use_batch_norm: bool = True, dropout_rate: float = 0.0,
                 activation: str = 'relu'):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        
        # Batch normalization
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.bn2 = nn.BatchNorm3d(out_channels)
            
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        # Dropout
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
            
        # Second conv block
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
            
        return x


class EncoderBlock3D(nn.Module):
    """Encoder block: Conv -> Pool, returns both conv output (for skip) and pooled output."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, 
                 use_batch_norm=True, dropout_rate=0.0, activation='relu'):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, kernel_size, padding, 
                               use_batch_norm, dropout_rate, activation)
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        conv_out = self.conv(x)      # For skip connection
        pooled_out = self.pool(conv_out)  # For next layer
        return pooled_out, conv_out


class DecoderBlock3D(nn.Module):
    """Decoder block: Upsample -> Concat with Skip -> Conv."""
    
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3, padding=1,
                 use_batch_norm=True, dropout_rate=0.0, activation='relu'):
        super().__init__()
        # ConvTranspose3d for upsampling (2x in all spatial dimensions)
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels//2, 2, stride=2)
        
        # Conv block after concatenation (in_channels//2 + skip_channels -> out_channels)
        concat_channels = in_channels//2 + skip_channels
        self.conv = ConvBlock3D(concat_channels, out_channels, kernel_size, padding,
                               use_batch_norm, dropout_rate, activation)
        
    def forward(self, x, skip):
        # Upsample
        x_up = self.upsample(x)
        
        # Concatenate with skip connection
        x_concat = torch.cat([x_up, skip], dim=1)
        
        # Apply convolution
        x_out = self.conv(x_concat)
        
        return x_out


class UNet3D(nn.Module):
    """Complete 3D U-Net with configurable depth and parameters."""
    
    def __init__(self, in_channels=8, out_channels=1, base_channels=16, depth=3,
                 kernel_size=3, padding=1, use_batch_norm=True, dropout_rate=0.0,
                 activation='relu', final_activation=None):
        super().__init__()
        
        self.depth = depth
        self.final_activation_name = final_activation
        
        # Validate input dimensions will be divisible by 2^depth
        # This validation will be done at forward time since we don't know input size here
        
        # Channel progression: base_channels * (2^i) for each layer
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Encoder path
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(
                EncoderBlock3D(channels[i], channels[i+1], kernel_size, padding,
                              use_batch_norm, dropout_rate, activation)
            )
            
        # Bottleneck
        bottleneck_in = channels[depth]
        bottleneck_out = channels[depth + 1]
        self.bottleneck = ConvBlock3D(bottleneck_in, bottleneck_out, kernel_size, padding,
                                     use_batch_norm, dropout_rate, activation)
        
        # Decoder path
        self.decoders = nn.ModuleList()
        for i in range(depth):
            decoder_in = channels[depth + 1 - i]  # Start from bottleneck output
            skip_channels = channels[depth - i]   # Skip connection channels
            decoder_out = channels[depth - i]     # Output channels match skip
            
            self.decoders.append(
                DecoderBlock3D(decoder_in, skip_channels, decoder_out, kernel_size, padding,
                              use_batch_norm, dropout_rate, activation)
            )
        
        # Final output layer
        self.final_conv = nn.Conv3d(channels[1], out_channels, 1)
        
        # Final activation
        if final_activation == 'relu':
            self.final_activation = nn.ReLU()
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = None
        
    def forward(self, x):
        # Validate input dimensions are divisible by 2^depth
        batch, channels, *spatial_dims = x.shape
        for i, dim_size in enumerate(spatial_dims):
            if dim_size % (2 ** self.depth) != 0:
                raise ValueError(f"Spatial dimension {i} (size {dim_size}) is not divisible by {2**self.depth}. "
                               f"All spatial dimensions must be divisible by 2^depth={2**self.depth} for depth={self.depth}.")
        
        # Encoder path
        skip_connections = []
        current = x
        
        for encoder in self.encoders:
            current, skip = encoder(current)
            skip_connections.append(skip)
            
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoder path (use reversed skip connections)
        for i, decoder in enumerate(self.decoders):
            skip_idx = self.depth - 1 - i  # Use skip connections in reverse order
            current = decoder(current, skip_connections[skip_idx])
        
        # Final output
        output = self.final_conv(current)
        
        # Apply final activation if specified
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output
    
    def get_architecture_info(self):
        """Get detailed architecture information."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate channel progression based on current configuration
        in_channels = next(iter(self.encoders[0].conv.conv1.parameters())).shape[1]
        base_channels = self.encoders[0].conv.conv1.out_channels
        channel_progression = [in_channels] + [base_channels * (2 ** i) for i in range(self.depth + 1)]
        
        return {
            'total_parameters': total_params,
            'depth': self.depth,
            'encoder_layers': self.depth,
            'decoder_layers': self.depth,
            'channel_progression': channel_progression,
            'final_activation': self.final_activation_name
        }


if __name__ == "__main__":
    # Test configurations
    test_configs = [
        {"depth": 3, "base_channels": 16, "input_shape": (2, 8, 32, 64, 16)},
        {"depth": 2, "base_channels": 32, "input_shape": (2, 8, 16, 32, 8)},
        {"depth": 4, "base_channels": 8, "input_shape": (2, 8, 64, 128, 32)}
    ]
    
    print("Testing Configurable 3D U-Net...")
    print("=" * 70)
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: depth={config['depth']}, base_channels={config['base_channels']}")
        print("-" * 50)
        
        test_input = torch.randn(*config["input_shape"])
        
        model = UNet3D(
            in_channels=8, 
            out_channels=1, 
            depth=config["depth"],
            base_channels=config["base_channels"],
            kernel_size=3,
            padding=1,
            use_batch_norm=True,
            dropout_rate=0.1,
            activation='relu',
            final_activation=None
        )
        
        try:
            # Test forward pass
            print(f"Input shape: {test_input.shape}")
            
            with torch.no_grad():  # Don't need gradients for shape testing
                output = model(test_input)
            
            print(f"Output shape: {output.shape}")
            
            # Verify dimensions
            expected_shape = (test_input.shape[0], 1) + test_input.shape[2:]
            if output.shape == expected_shape:
                print("✅ Output shape matches expected dimensions!")
            else:
                print(f"❌ Shape mismatch! Expected: {expected_shape}, Got: {output.shape}")
                
            # Architecture info
            info = model.get_architecture_info()
            print(f"📊 Architecture Info:")
            print(f"  Total parameters: {info['total_parameters']:,}")
            print(f"  Depth: {info['depth']}")
            print(f"  Channel progression: {info['channel_progression']}")
            print(f"  Final activation: {info['final_activation']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test error handling for non-divisible dimensions
    print("\n" + "=" * 70)
    print("Testing Error Handling for Non-Divisible Dimensions...")
    
    try:
        bad_input = torch.randn(2, 8, 31, 63, 15)  # Not divisible by 8 (2^3)
        model = UNet3D(depth=3)
        output = model(bad_input)
        print("❌ Should have raised an error!")
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")