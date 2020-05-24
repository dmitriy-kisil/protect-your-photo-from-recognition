import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from flask import Flask
import os
import base64
import cv2
import numpy as np
import flask
import json
import tensorflow as tf
from utils import preprocess, get_imagenet_label, create_adversarial_pattern
stylesheets = [
    "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.2/css/bulma.min.css", # Bulma
]

server = Flask(__name__)
# create app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=stylesheets
)


app.layout = html.Div(
    className="section",
    children=[
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        html.Label('Epsilon'),
        dcc.Slider(id='epsilon', min=0.01, max=0.1, step=0.01,
                   marks=dict(zip([i for i in np.arange(0.01, 0.11, 0.01)],
                                  [str(round(i, 3)) for i in np.arange(0.01, 0.11, 0.01)])),
                   value=0.01, included=False
                   ),
        html.Div([html.Label('original image', style={'width': '33%', 'display': 'inline-block'}),
                  html.Div(id='show-choosed-epsilon-text', children=[], style={'width': '33%', 'display': 'inline-block'}),
                  html.Label('After FGSM', style={'width': '33%', 'display': 'inline-block'})]),
        html.Div([html.Div(id='original-prediction', children=[], style={'width': '33%', 'display': 'inline-block'}),
                  html.Div(id="download-area", children=[], style={'width': '33%', 'display': 'inline-block'}),
                  html.Div(id='changed-prediction', children=[], style={'width': '33%', 'display': 'inline-block'}),
                  ]),
        html.Div([html.Div(id='original-image', children=[], style={'width': '33%', 'display': 'inline-block'}),
                  html.Div(id='perturbations', children=[], style={'width': '33%', 'display': 'inline-block'}),
                  html.Div(id='changed-image', children=[], style={'width': '33%', 'display': 'inline-block'})])
    ]
)


def build_download_button(uri):
    """Generates a download button for the resource"""
    button = html.Form(
        action=uri,
        method="get",
        children=[
            html.Button(
                className="button",
                type="submit",
                children=[
                    "download"
                ]
            )
        ]
    )
    return button


@app.callback(
    Output(component_id='original-image', component_property='children'),
    [Input('upload-image', 'contents')]
)
def update_original_image(list_of_contents):

    if list_of_contents is None:
        return html.Img(id='image')
    data = list_of_contents[-1].encode("utf8").split(b";base64,")[1]
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    buffer = cv2.imencode('.jpg', img)[1]

    encoded_image = base64.b64encode(buffer)

    return html.Img(id='image', src='data:image/png;base64,{}'.format(
        encoded_image.decode()))


@app.callback(
    Output(component_id='perturbations', component_property='children'),
    [Input('epsilon', 'value'), Input('original-image', 'children')]
)
def update_perturbations(epsilon_value, img_src):
    if not img_src['props'].get('src'):
        return
    data = img_src['props']['src'].split(";base64,")[1]
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # machine learning part
    image_raw = np.asarray(img).astype(np.float32)
    image = tf.convert_to_tensor(image_raw, tf.float32)
    image = preprocess(image)
    image_probs = pretrained_model.predict(image)
    _, label, confidence = get_imagenet_label(image_probs)
    for index, class_name in data_json.items():
        if label in class_name:
            class_index = int(index)
            break
    # Get the input label of the image.

    label = tf.one_hot(class_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturbations = create_adversarial_pattern(image, label)

    image = perturbations[0] * 0.5 + 0.5
    image_to_save = tf.squeeze(tf.image.convert_image_dtype(image, tf.uint8))
    image_to_save = image_to_save.numpy()
    buffer = cv2.imencode('.jpg', image_to_save)[1]

    encoded_image = base64.b64encode(buffer)

    return html.Img(id='image', src='data:image/png;base64,{}'.format(
        encoded_image.decode()))


@app.callback(
    Output(component_id='changed-image', component_property='children'),
    [Input('epsilon', 'value'), Input('original-image', 'children'), Input('perturbations', 'children')]
)
def update_changed_image(epsilon_value, orig_img_src, perturb_img_src):
    if not orig_img_src['props'].get('src'):
        return
    orig_data = orig_img_src['props']['src'].split(";base64,")[1]
    orig_nparr = np.frombuffer(base64.b64decode(orig_data), np.uint8)
    orig_img = cv2.imdecode(orig_nparr, cv2.IMREAD_COLOR)
    # machine learning part
    orig_image_raw = np.asarray(orig_img).astype(np.float32)
    orig_image = tf.convert_to_tensor(orig_image_raw, tf.float32)
    orig_image = preprocess(orig_image)

    # perturb_data = perturb_img_src['props']['src'].split(";base64,")[1]
    # perturb_nparr = np.frombuffer(base64.b64decode(perturb_data), np.uint8)
    # perturb_img = cv2.imdecode(perturb_nparr, cv2.IMREAD_COLOR)
    # # machine learning part
    # perturb_image_raw = np.asarray(perturb_img).astype(np.float32)
    # perturb_image = tf.convert_to_tensor(perturb_image_raw, tf.float32)
    image_probs = pretrained_model.predict(orig_image)
    _, label, confidence = get_imagenet_label(image_probs)
    for index, class_name in data_json.items():
        if label in class_name:
            class_index = int(index)
            break

    # Get the input label of the image.

    label = tf.one_hot(class_index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    perturb_image = create_adversarial_pattern(orig_image, label)
    perturb_image = perturb_image[None, ...]
    adv_x = orig_image + epsilon_value * perturb_image
    image = tf.clip_by_value(adv_x, -1, 1)
    image = image[0] * 0.5 + 0.5
    image_to_save = tf.squeeze(tf.image.convert_image_dtype(image, tf.uint8))
    image_to_save = image_to_save.numpy()
    buffer = cv2.imencode('.jpg', image_to_save)[1]

    encoded_image = base64.b64encode(buffer)

    return html.Img(id='image', src='data:image/png;base64,{}'.format(
        encoded_image.decode()))


@app.callback(
    Output(component_id='show-choosed-epsilon-text', component_property='children'),
    [Input('epsilon', 'value')]
)
def update_choosed_epsilon_text(epsilon_value):
    return f"Selected epsilon value: {epsilon_value}"


@app.callback(
    Output(component_id='original-prediction', component_property='children'),
    [Input('original-image', 'children')]
)
def update_original_predictions(img_src):
    if not img_src['props'].get('src'):
        return
    data = img_src['props']['src'].split(";base64,")[1]
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_raw = np.asarray(img).astype(np.float32)
    image = tf.convert_to_tensor(image_raw, np.float32)
    image = preprocess(image)
    _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
    return '{} : {:.2f}% Confidence'.format(label, confidence*100)


@app.callback(
    Output(component_id='changed-prediction', component_property='children'),
    [Input('changed-image', 'children')]
)
def update_changed_predictions(img_src):
    if not img_src:
        return
    data = img_src['props']['src'].split(";base64,")[1]
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("downloadable/example2.jpg", img)
    image_raw = np.asarray(img).astype(np.float32)
    image = tf.convert_to_tensor(image_raw, np.float32)

    image = preprocess(image)
    image_to_save = tf.squeeze(tf.image.convert_image_dtype(image, tf.uint8))
    image_to_save = cv2.cvtColor(image_to_save.numpy(), cv2.COLOR_RGB2BGR)
    cv2.imwrite("downloadable/example3.jpg", image_to_save)
    image_probs = pretrained_model.predict(image)
    _, label, confidence = get_imagenet_label(image_probs)
    return '{} : {:.2f}% Confidence'.format(label, confidence*100)


@app.callback(
    Output("download-area", "children"),
    [Input('upload-image', 'filename'), Input('changed-image', 'children')],
)
def show_download_button(filename, img_src):
    if not img_src:
        return
    data = img_src['props']['src'].split(";base64,")[1]
    nparr = np.frombuffer(base64.b64decode(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    path = f"downloadable/test_{filename[0]}"
    cv2.imwrite(path, img)
    uri = path
    return [build_download_button(uri)]


@app.server.route('/downloadable/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(
        os.path.join(root_dir, 'downloadable'), path
    )


if __name__ == '__main__':
    pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                         weights='imagenet')
    pretrained_model.trainable = False
    imagenet_labels = 'imagenet_class_index.json'
    with open(imagenet_labels) as json_file:
        data_json = json.load(json_file)

    app.run_server(host="0.0.0.0", port="8050", debug=True)
