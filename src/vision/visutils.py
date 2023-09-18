import layoutparser as lp


MODELS = {
    "navigator": {
        "model": 'lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config',
        "labels": {0: "Photograph", 1: "Illustration", 2: "Map", 3: "Comics/Cartoon", 4: "Editorial Cartoon", 5: "Headline", 6: "Advertisement"}
    },
    "hjdataset": {
        "model": "lp://HJDataset/faster_rcnn_R_50_FPN_3x/config",
        "labels": {1:"Page Frame", 2:"Row", 3:"Title Region", 4:"Text Region", 5:"Title", 6:"Subtitle", 7:"Other"}
    },
    "prima": {
        "model": "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
        "labels": {1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}
    }
}  

def load_model(name, device = 'cpu'):
    model = lp.models.Detectron2LayoutModel(
        config_path =MODELS[name]['model'], # In model catalog
        label_map   = MODELS[name]["labels"], # In model`label_map`
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8] # Optional
    )
    model.device = device
    return model