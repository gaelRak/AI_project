import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("model/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("model/val.csv").values.tolist()

    accum_cer = []
    i=0
    image_path = "image/created/ROI.png"
    image = cv2.imread(image_path)
    label = input("Entrer label corespondent:")
    prediction_text = model.predict(image)

    cer = get_cer(prediction_text, label)
    print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
    image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''for image_path, label in tqdm(df):
        #image = cv2.imread(image_path)
        image_path="drawn_image.png"
        image= cv2.imread(image_path)
        pred=input("Entrer la prediction corespandente")
        prediction_text = model.predict(image)

        cer = get_cer(pred, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

        # resize by 2x
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        i+=1'''

    print(f"Average CER: {np.average(accum_cer)}")