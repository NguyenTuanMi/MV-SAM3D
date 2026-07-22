"""
SAM3 多物体分割模块
"""
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image
from loguru import logger


class SAM3MultiObjectSegmenter:
    """SAM3 多物体分割器"""
    
    def __init__(self, checkpoint_path: Path = None, confidence_threshold: float = 0.1):
        """
        初始化 SAM3 模型
        
        Args:
            checkpoint_path: SAM3 checkpoint 路径
            confidence_threshold: 置信度阈值
        """
        if checkpoint_path is None:
            checkpoint_path = Path("/mnt/workspace/users/lbc/sam3/checkpoints/sam3.pt")
        
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        
        # 添加 sam3 路径
        sam3_path = Path("/mnt/workspace/users/lbc/sam3")
        if sam3_path not in [Path(p) for p in sys.path]:
            sys.path.insert(0, str(sam3_path))
        
        # 导入 SAM3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        # 加载模型
        logger.info(f"Loading SAM3 model from: {checkpoint_path}")
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
        logger.success("✓ SAM3 model loaded")
    
    def segment_object_multiview(
        self,
        images_dir: Path,
        object_name: str,
        text_prompt: str,
        output_dir: Path,
    ) -> Dict:
        """
        对多个视角的图像分割同一个物体
        
        Args:
            images_dir: 图像目录
            object_name: 物体名称
            text_prompt: SAM3 文本提示词
            output_dir: 输出目录（将创建 object_name/ 子目录）
        
        Returns:
            Dict with status and info
        """
        logger.info(f"\n[Segmenting: {object_name}]")
        logger.info(f"  Prompt: '{text_prompt}'")
        
        # 获取所有图像（使用自然数字排序，确保 2.png 排在 10.png 前面）
        def natural_sort_key(p):
            try:
                return (0, int(p.stem), p.stem)
            except ValueError:
                return (1, 0, p.stem)
        
        image_files = sorted(
            list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")),
            key=natural_sort_key
        )
        if not image_files:
            return {'success': False, 'error': 'No images found'}
        
        logger.info(f"  Processing {len(image_files)} views...")
        
        # 创建物体的 mask 目录
        mask_dir = output_dir / object_name
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # 对每个视角进行分割
        success_count = 0
        failed_views = []
        
        for i, img_path in enumerate(image_files):
            try:
                # 读取图像
                image = Image.open(img_path).convert('RGB')
                
                # SAM3 分割（按照SAM4D的方式）
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inference_state = self.processor.set_image(image)
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=text_prompt
                    )
                
                masks = output["masks"]
                scores = output["scores"]
                
                if len(masks) == 0:
                    logger.warning(f"  View {i}: No mask generated (confidence too low)")
                    failed_views.append(i)
                    continue
                
                # 选择最高分的 mask
                best_idx = scores.argmax().item()
                best_mask = masks[best_idx]
                best_score = scores[best_idx].item()
                
                # 转换为 numpy（按照SAM4D的方式）
                mask_np = best_mask.squeeze(0).cpu().numpy() if torch.is_tensor(best_mask) else best_mask.squeeze(0)
                
                # 确保 mask 和原图尺寸一致
                if mask_np.shape != (image.size[1], image.size[0]):
                    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize(image.size, Image.NEAREST)
                    mask_np = np.array(mask_pil) / 255.0
                
                # 创建 RGBA 格式的 mask（与 example 一致）
                # RGB 保留原图颜色，Alpha 作为 mask
                image_np = np.array(image)
                mask_bool = mask_np > 0.5
                
                # 创建 RGBA 图像
                rgba_mask = np.zeros((image_np.shape[0], image_np.shape[1], 4), dtype=np.uint8)
                # mask 区域：保留原图 RGB，alpha=255
                rgba_mask[mask_bool, :3] = image_np[mask_bool]
                rgba_mask[mask_bool, 3] = 255
                # 非 mask 区域：RGB=0，alpha=0（透明）
                rgba_mask[~mask_bool, :3] = 0
                rgba_mask[~mask_bool, 3] = 0
                
                # 保存为 RGBA PNG（使用原图文件名，确保mask和image名称一致）
                mask_path = mask_dir / f"{img_path.stem}.png"
                Image.fromarray(rgba_mask, 'RGBA').save(mask_path)
                
                # 计算 mask 面积（使用 alpha 通道）
                area_ratio = np.sum(rgba_mask[:, :, 3] > 0) / (rgba_mask.shape[0] * rgba_mask.shape[1])
                
                logger.info(f"  View {i}: ✓ (area={area_ratio*100:.1f}%, score={best_score:.3f})")
                success_count += 1
                
            except Exception as e:
                logger.error(f"  View {i}: Failed - {e}")
                import traceback
                traceback.print_exc()
                failed_views.append(i)
        
        # 总结
        logger.info(f"  Result: {success_count}/{len(image_files)} views segmented")
        if failed_views:
            logger.warning(f"  Failed views: {failed_views}")
        
        return {
            'success': success_count > 0,
            'object_name': object_name,
            'total_views': len(image_files),
            'success_views': success_count,
            'failed_views': failed_views,
            'mask_dir': mask_dir,
        }
