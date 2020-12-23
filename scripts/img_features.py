import json

import detectron2
from more_itertools import chunked
from itertools import islice
from typing import Callable, List, Tuple, TypedDict
import torch
import detectron2 as dt
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputs,
    fast_rcnn_inference,
    fast_rcnn_inference_single_image,
)
from detectron2.modeling.poolers import ROIPooler
import detectron2.data.transforms as T
from detectron2.modeling.postprocessing import detector_postprocess

from detectron2 import model_zoo
from detectron2.modeling import GeneralizedRCNN
from detectron2.utils.visualizer import Visualizer
import numpy as np
import cv2  # type: ignore
from torch.nn.modules.module import Module
from torch.tensor import Tensor
import tqdm
import typer
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image  # type: ignore
from dnips.data.loading import PerFileDataset
from torchvision import models, transforms  # type: ignore
import torch


import os
import io

##


app = typer.Typer()


@app.command()
def extract_resnet(
    img_d: Path,
    out_d: Path,
    resnet_ver: str = "resnet101",
    b_size: int = 128,
    workers: int = 20,
) -> None:

    use_cuda = torch.cuda.is_available()
    print(f"Loading model ...")
    resnet = models.resnet101(pretrained=True)

    # All but the last layer
    modules = list(resnet.children())[:-1]
    model: "Module" = torch.nn.Sequential(*modules)

    # Resnet prep
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def read_func(fpath: Path) -> Tensor:
        image = Image.open(fpath).convert("RGB")
        input_tensor = preprocess(image)
        return input_tensor

    torch.no_grad()
    model.eval()
    if use_cuda:
        model.to("cuda")

    fpaths = list(img_d.iterdir())
    dataset = PerFileDataset(fpaths, read_func)
    dloader: "DataLoader[Tensor, Tensor]" = DataLoader(
        dataset, batch_size=b_size, shuffle=False, num_workers=workers
    )

    all_features = []
    for batch in tqdm.tqdm(dloader):
        if use_cuda:
            batch = batch.cuda()
        features = model(batch)
        print(features.size())
        all_features.append(features)

    # final_feats = torch.cat(all

    breakpoint()


##
class Result(TypedDict):
    instances: Instances
    roi_features: Tensor


class BatchPredictor:
    """Modified from detectron2.engine.DefaultPredictor

    Borrows a LOT of code
        from: https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/demo_feature_extraction.ipynb
    """

    # This is the default in  detectron2.modeling.roi_heads.fast_rcnn.FastRCNNOutputs
    NMS_THRESH = 0.5

    def __init__(self, cfg, score_thresh: float = 0.6):
        self.score_thresh = score_thresh
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model: GeneralizedRCNN = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images: List[np.ndarray]) -> List[Result]:
        """
        Args:
            original_images (np.ndarray): A batch of images of shape (H, W, C) (in
            BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        if len(original_images) == 0:
            return []
        assert original_images[0].ndim == 3
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_images = [im[:, :, ::-1] for im in original_images]

            ######### pass through backbone: batched computation (i hope) #####
            images = []
            heights = []
            widths = []
            for im in original_images:
                # get_transform doesn't take a batch
                tsfm = self.aug.get_transform(im)

                # apply_image takes a batch
                images.append(tsfm.apply_image(im))
                heights.append(im.shape[0])
                widths.append(im.shape[1])

            images = [
                torch.from_numpy(im.astype(np.float32).transpose((2, 0, 1)))
                for im in images
            ]

            inputs = [
                {"image": im, "height": h, "width": w}
                for im, h, w in zip(images, heights, widths)
            ]
            prepped_inputs = self.model.preprocess_image(inputs)
            # run C1 through C4
            features = self.model.backbone(prepped_inputs.tensor)

            ######## RPN proposals. Definitely not batched #############
            proposals, _ = self.model.proposal_generator(prepped_inputs, features, None)

            results = []
            for proposal, height, width in zip(proposals, heights, widths):

                print("Proposal Boxes size:", proposal.proposal_boxes.tensor.shape)

                # Run RoI head for each proposal (RoI Pooling + Res5)
                proposal_boxes = [x.proposal_boxes for x in proposals]
                features: Tensor = [features[f] for f in self.model.roi_heads.in_features]  # type: ignore
                box_features: Tensor = self.model.roi_heads._shared_roi_transform(  # type: ignore
                    features, proposal_boxes
                )
                feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
                print("Pooled features size:", feature_pooled.shape)

                # Predict classes and boxes for each proposal.
                pred_class_logits: Tensor
                pred_proposal_deltas: Tensor
                (
                    pred_class_logits,
                    pred_proposal_deltas,
                ) = self.model.roi_heads.box_predictor(
                    feature_pooled
                )  # type: ignore

                outputs = FastRCNNOutputs(
                    self.model.roi_heads.box2box_transform,
                    pred_class_logits,
                    pred_proposal_deltas,
                    proposals,
                    self.model.roi_heads.smooth_l1_beta,
                )
                probs = outputs.predict_probs()[0]
                boxes = outputs.predict_boxes()[0]

                # Note: BUTD uses raw RoI predictions,
                #       we use the predicted boxes instead.
                # boxes = proposal_boxes[0].tensor

                # NMS
                instances, ids = fast_rcnn_inference_single_image(
                    boxes,
                    probs,
                    images[0].shape[1:],  # type: ignore[arg-type]
                    score_thresh=self.score_thresh,
                    nms_thresh=self.NMS_THRESH,
                    topk_per_image=-1,  # Return all above score threshold
                )

                instances = detector_postprocess(instances, height, width)
                roi_features = feature_pooled[ids].detach()
                print(instances)

                results.append(Result(instances=instances, roi_features=roi_features))

            return results
##


@app.command()
def ext_fRCNN(img_d: Path, out_d: Path, resnet_ver: str,) -> None:

    ##
    # Load VG Classes
    vg_classes = []
    with Path("objects_vocab.txt").open() as f:
        for obj in f.readlines():
            vg_classes.append(obj.split(",")[0].lower().strip())

    MetadataCatalog.get("vg").thing_classes = vg_classes
    ##

    img_d = Path("/projectnb/llamagrp/davidat/datasets/flickr30k/flickr30k_images/")
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
"COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"
    )

    predictor = BatchPredictor(cfg)

    ##


    ##

    all_fpaths = list(img_d.iterdir())[:100]

    ## Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    for fpaths in chunked(all_fpaths, 5):

        ims = [cv2.imread(str(fp)) for fp in fpaths]

        outputs = predictor(ims)

        for im, output, fpath in zip(ims, outputs, fpaths):
            v = Visualizer(
                im[:, :, ::-1],
                MetadataCatalog.get(
                    cfg.DATASETS.TRAIN[0]  # type: ignore[misc]
                ),
                scale=1.2,
            )
            out = v.draw_instance_predictions(output["instances"].to("cpu"))

            img = out.get_image()[:, :, ::-1]
            cv2.imwrite(fpath.name, img)
    ##


if __name__ == "__main__":
    app()
