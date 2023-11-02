import torch
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
)
from transformers.image_transforms import center_to_corners_format
from transformers.models.owlvit.modeling_owlvit import box_iou
from functools import partial
import numpy as np
from transformers.models.owlvit.modeling_owlvit import (
    OwlViTImageGuidedObjectDetectionOutput,
)


processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")


def classpredictionhead_box_forward(
    self,
    image_embeds,
    query_indice,
    query_mask,
):
    image_class_embeds = self.dense0(image_embeds)

    # Normalize image and text features
    image_class_embeds = image_class_embeds / (
        torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
    )
    query_embeds = image_class_embeds[0, query_indice].unsqueeze(0).unsqueeze(0)
    # query_embeds = query_embeds / (torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6)

    # Get class predictions
    pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

    # Apply a learnable shift and scale to logits
    logit_shift = self.logit_shift(image_embeds)
    logit_scale = self.logit_scale(image_embeds)
    logit_scale = self.elu(logit_scale) + 1
    pred_logits = (pred_logits + logit_shift) * logit_scale

    if query_mask is not None:
        if query_mask.ndim > 1:
            query_mask = torch.unsqueeze(query_mask, dim=-2)

        pred_logits = pred_logits.to(torch.float64)
        pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
        pred_logits = pred_logits.to(torch.float32)

    return (pred_logits, image_class_embeds)


def class_predictor(
    self,
    image_feats,
    query_indice=None,
    query_mask=None,
):
    (pred_logits, image_class_embeds) = self.class_head.classpredictionhead_box_forward(
        image_feats, query_indice, query_mask
    )

    return (pred_logits, image_class_embeds)


def get_max_iou_indice(target_pred_boxes, query_box, target_sizes):
    boxes = center_to_corners_format(target_pred_boxes)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]

    iou, _ = box_iou(boxes.squeeze(0), query_box)

    return iou.argmax()


def box_guided_detection(
    self: OwlViTForObjectDetection,
    pixel_values,
    query_box=None,
    target_sizes=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.return_dict

    # Compute feature maps for the input and query images
    # query_feature_map = self.image_embedder(pixel_values=query_pixel_values)[0]
    feature_map, vision_outputs = self.image_embedder(
        pixel_values=pixel_values,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
    image_feats = torch.reshape(
        feature_map, (batch_size, num_patches * num_patches, hidden_dim)
    )

    # batch_size, num_patches, num_patches, hidden_dim = query_feature_map.shape
    # query_image_feats = torch.reshape(query_feature_map, (batch_size, num_patches * num_patches, hidden_dim))
    # # Get top class embedding and best box index for each query image in batch
    # query_embeds, best_box_indices, query_pred_boxes = self.embed_image_query(query_image_feats, query_feature_map)

    # Predict object boxes
    target_pred_boxes = self.box_predictor(image_feats, feature_map)

    # Get MAX IOU box corresponding embedding
    query_indice = get_max_iou_indice(target_pred_boxes, query_box, target_sizes)

    # Predict object classes [batch_size, num_patches, num_queries+1]
    (pred_logits, class_embeds) = self.class_predictor(
        image_feats=image_feats, query_indice=query_indice
    )

    if not return_dict:
        output = (
            feature_map,
            # query_feature_map,
            target_pred_boxes,
            # query_pred_boxes,
            pred_logits,
            class_embeds,
            vision_outputs.to_tuple(),
        )
        output = tuple(x for x in output if x is not None)
        return output

    return OwlViTImageGuidedObjectDetectionOutput(
        image_embeds=feature_map,
        # query_image_embeds=query_feature_map,
        target_pred_boxes=target_pred_boxes,
        # query_pred_boxes=query_pred_boxes,
        logits=pred_logits,
        class_embeds=class_embeds,
        text_model_output=None,
        vision_model_output=vision_outputs,
    )


model.box_guided_detection = partial(box_guided_detection, model)
model.class_predictor = partial(class_predictor, model)
model.class_head.classpredictionhead_box_forward = partial(
    classpredictionhead_box_forward, model.class_head
)
