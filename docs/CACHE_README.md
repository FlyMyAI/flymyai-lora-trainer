# Quantized Model Cache System

This document describes the cache system implemented for quantized models to avoid re-quantization on every training run.

## Overview

The cache system automatically saves quantized models after the first quantization run and loads them from cache on subsequent runs, significantly reducing training startup time.

## Features

- **Automatic Caching**: Quantized models are automatically saved after quantization
- **Smart Cache Keys**: Cache keys are generated based on model path, dtype, and LoRA rank
- **Integrity Validation**: Cache files are validated before loading to prevent corruption issues
- **Disk Space Management**: Automatic cleanup of old and corrupted cache files
- **Environment Variable Configuration**: Uses `QWEN_CACHE` environment variable for cache location

## Usage

### Basic Usage

The cache system is automatically enabled when `args.quantize=True`. No additional configuration is needed.

### Environment Variable

Set the `QWEN_CACHE` environment variable to specify a custom cache directory:

```bash
export QWEN_CACHE="/path/to/your/cache"
```

If not set, the default cache directory is `~/.cache/qwen_quantized`.

### Cache Behavior

1. **First Run**: Model is quantized and saved to cache
2. **Subsequent Runs**: Model is loaded from cache (much faster)
3. **Configuration Changes**: If model path, dtype, or rank changes, a new cache entry is created

## Cache Functions

### Core Functions

- `get_quantized_model_cache_path(model_path, weight_dtype, rank)`: Generate cache path
- `save_quantized_model(model, cache_path)`: Save model to cache
- `load_quantized_model(model, cache_path)`: Load model from cache
- `validate_cache_integrity(cache_path)`: Validate cache file integrity

### Management Functions

- `get_cache_info(cache_dir)`: Get cache directory information
- `clear_old_cache_files(cache_dir, max_age_days=7)`: Remove old cache files
- `cleanup_corrupted_cache(cache_dir)`: Remove corrupted cache files

## Cache Key Generation

Cache keys are generated using MD5 hash of:
```
{model_path}_{weight_dtype}_{rank}
```

This ensures that different configurations get separate cache entries.

## Automatic Cleanup

The system automatically:
- Validates cache integrity on load
- Removes corrupted cache files
- Logs cache statistics and disk space information
- Warns about low disk space

## Testing

Run the test script to verify the cache system:

```bash
python test_cache.py
```

## Troubleshooting

### Cache Not Working

1. Check if `args.quantize=True` is set
2. Verify the `QWEN_CACHE` environment variable is set correctly
3. Check disk space availability
4. Look for cache-related log messages

### Corrupted Cache

The system automatically detects and removes corrupted cache files. If you encounter issues:

1. Clear the cache directory manually
2. Check disk space and permissions
3. Verify the model configuration hasn't changed

### Performance Issues

1. Ensure cache directory is on a fast storage device (SSD recommended)
2. Monitor disk I/O during cache operations
3. Consider using a RAM disk for very fast access

## Logging

The cache system provides detailed logging:
- Cache directory information
- Save/load success/failure
- Cache validation results
- Disk space warnings
- Cleanup operations

## Example Log Output

```
INFO - Using quantized model cache directory: /path/to/cache
INFO - Cache info: {'cache_dir': '/path/to/cache', 'file_count': 2, 'total_size_mb': 45.2}
INFO - Available disk space: 125.67 GB
INFO - Successfully loaded quantized model from cache
```

## Configuration

The cache system respects the following configuration:
- `args.quantize`: Enable/disable quantization and caching
- `args.pretrained_model_name_or_path`: Model path for cache key
- `args.rank`: LoRA rank for cache key
- `weight_dtype`: Model dtype for cache key

## Best Practices

1. **Use SSD Storage**: Place cache directory on fast storage
2. **Monitor Disk Space**: Regular cleanup prevents disk space issues
3. **Version Control**: Consider backing up important cache files
4. **Network Storage**: For shared environments, use network storage for cache
5. **Regular Testing**: Test cache functionality after configuration changes