export CUDA_VISIBLE_DEVICES=2
python scripts/evaluate.py  --model_path /mnt/home/wudidi/code_v5/qwen3-vl-det/model_cache/Qwen/Qwen3-VL-2B-Instruct   \
                            --test_data /mnt/home/wudidi/code_v5/qwen3-vl-det/data/hefei_last_dataset/qwen_data/test.json     \
                            --output_dir eval_results --iou_threshold 0.5