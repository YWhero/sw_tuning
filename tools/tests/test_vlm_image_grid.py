"""Regression: legacy rectangular camera crops must survive Qwen patchification."""
import unittest
import torch
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator


class ImageGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collator = Gr00tN1d7DataCollator(
            'nvidia/Cosmos-Reason2-2B', transformers_loading_kwargs={'local_files_only': True})

    def feature(self, height, width):
        image = torch.arange(3*height*width).remainder(256).to(torch.uint8).reshape(3,height,width)
        p = self.collator.processor
        text = p.apply_chat_template([{'role':'user','content':[
            {'type':'image','image':image}, {'type':'text','text':'Clean the table.'}]}],
            tokenize=False, add_generation_prompt=False)
        return {'vlm_content': {'text':text,'images':[image]}}

    def test_rectangular_crops_patchify(self):
        for height,width in [(256,340),(256,455),(340,256)]:
            with self.subTest(size=(height,width)):
                output = self.collator([self.feature(height,width)])["inputs"]
                self.assertTrue(torch.isfinite(output['pixel_values']).all())
                self.assertEqual(output['pixel_values'].shape[0],output['image_grid_thw'].prod().item())

    def test_unpadded_multiview_images_and_masks(self):
        import numpy as np
        from PIL import Image
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
        from gr00t.model.gr00t_n1d7.image_augmentations import build_image_transformations_albumentations
        processor = Gr00tN1d7Processor.__new__(Gr00tN1d7Processor)
        processor.processor = self.collator.processor
        processor.use_albumentations = True
        processor.emit_attn_targets = True
        transforms = build_image_transformations_albumentations(
            None, None, None, None, 256, .9, letter_box_transform=False)
        keys = ['head', 'wrist']
        images = {key: [Image.fromarray(np.full((h,w,3), 120, np.uint8))]
                  for key,h,w in [('head',180,320), ('wrist',320,240)]}
        masks = {key: [np.ones((image[0].height,image[0].width),np.uint8)]
                 for key,image in images.items()}
        for transform in transforms:
            feature = processor._get_vlm_inputs(keys, images, masks, transform, 'Pick bottle')
            self.assertEqual(len(feature['vlm_content']['images']), 2)
            for rgb, mask in zip(feature['vlm_content']['images'], feature['attn_masks']):
                self.assertEqual(tuple(rgb.shape[-2:]), mask.shape)
                self.assertNotEqual(*mask.shape)
                self.assertTrue(mask.all(), 'No black padding may be added')
            output = self.collator([feature])['inputs']
            for row, grid in zip(output['attn_targets'], output['image_grid_thw']):
                count = int(grid.prod()) // 4
                self.assertTrue(torch.equal(row[:count], torch.ones(count)))
                self.assertTrue((row[count:] == -1).all())

    def test_aligned_training_crop_unchanged(self):
        feature=self.feature(256,256)
        output=self.collator([feature])["inputs"]
        content=feature['vlm_content']
        expected=self.collator.processor(text=[content['text']],images=content['images'],
            return_tensors='pt',padding=True,do_resize=False)
        for key in ['pixel_values','image_grid_thw','input_ids']:
            self.assertTrue(torch.equal(output[key],expected[key]),key)

    def test_reproduces_previous_failure(self):
        content=self.feature(256,340)['vlm_content']
        with self.assertRaises(RuntimeError):
            self.collator.processor(text=[content['text']],images=content['images'],
                return_tensors='pt',padding=True,do_resize=False)

if __name__=='__main__':
    unittest.main()
