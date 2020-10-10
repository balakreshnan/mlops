# Hello World for GPT-2 Model using Azure Machine learning Service

## Create a hello world for OpenAI GPT-2 model

## Prerequisite

- Create a Azure Machine learning Service
- Go to Compute and create a compute instance
- Start the compute instance and once started click JupyterLab
- Now time to write your first open AI GPT-2 model

## Steps for Hello world for OpenAI GPT-2 model

- Let's first clone the repo

```
!git clone https://github.com/openai/gpt-2.git
```

- Let's now download the model

```
!python ./gpt-2/download_model.py 124M
```

- Connect to ML workspace to register the model

```
from azureml.core import Model, Workspace
from IPython.core.display import display, Markdown
from markdown import markdown

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
```

- Now register the model with sample model card

```
model = Model.register(workspace=ws,
                       model_name='gpt-2',                # Name of the registered model in your workspace.
                       model_path='./models/124M',  # Local folder
                       model_framework=Model.Framework.PYTORCH,  # Framework used to create the model.
                       model_framework_version='1.3',             # Version of PyTorch to create the model.
                       description='This model was developed by researchers at OpenAI to help us understand how the capabilities of language model capabilities scale as a function of the size of the models',
                       tags={'title': 'GPT-2 model card',
    'datasheet_description':
"""
Last updated: November 2019

Inspired by [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993), we’re providing some accompanying information about the GPT-2 family of models we're releasing.

""",
    'details': 'This model was developed by researchers at OpenAI to help us understand how the capabilities of language model capabilities scale as a function of the size of the models (by parameter count) combined with very large internet-scale datasets (WebText).',
    'date': 'February 2019, trained on data that cuts off at the end of 2017.', 
    'type': 'Language model',
    'version': '1.5 billion parameters: the fourth and largest GPT-2 version. We have also released 124 million, 355 million, and 774 million parameter models.',
    'help': 'https://forms.gle/A7WBSbTY2EkKdroPA',
    'usecase_primary': 
"""
The primary intended users of these models are *AI researchers and practitioners*.

We primarily imagine these language models will be used by researchers to better understand the behaviors, capabilities, biases, and                                 constraints of large-scale generative language models.
""",
    'usecase_secondary':
"""
Here are some secondary use cases we believe are likely:

- **Writing assistance**: Grammar assistance, autocompletion (for normal prose or code)
- **Creative writing and art**: exploring the generation of creative, fictional texts; aiding creation of poetry and other literary art.
- **Entertainment**: Creation of games, chat bots, and amusing generations.
""",
    'usecase_outofscope':
"""
Because large-scale language models like GPT-2 do not distinguish fact from fiction, we don’t support use-cases that require the generated text to be true.

Additionally, language models like GPT-2 reflect the biases inherent to the systems they were trained on, so we do not recommend that they be deployed into systems that interact with humans unless the deployers first carry out a study of biases relevant to the intended use-case. We found no statistically significant difference in gender, race, and religious bias probes between 774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of caution around use cases that are sensitive to biases around human attributes.
""",
    'dataset_description':
"""
This model was trained on (and evaluated against) WebText, a dataset consisting of the text contents of 45 million links posted by users of the ‘Reddit’ social network. WebText is made of data derived from outbound links from Reddit and does not consist of data taken directly from Reddit itself. Before generating the dataset we used a blocklist to ensure we didn’t sample from a variety of subreddits which contain sexually explicit or otherwise offensive content.

To get a sense of the data that went into GPT-2, we’ve [published a list](domains.txt) of the top 1,000 domains present in WebText and their frequency.  The top 15 domains by volume in WebText are: Google, Archive, Blogspot, GitHub, NYTimes, Wordpress, Washington Post, Wikia, BBC, The Guardian, eBay, Pastebin, CNN, Yahoo!, and the Huffington Post.
""",
    'motivation': 'The motivation behind WebText was to create an Internet-scale, heterogeneous dataset that we could use to test large-scale language models against. WebText was (and is) intended to be primarily for research purposes rather than production purposes.',
    'caveats':
"""
Because GPT-2 is an internet-scale language model, it’s currently difficult to know what disciplined testing procedures can be applied to it to fully understand its capabilities and how the data it is trained on influences its vast range of outputs. We recommend researchers investigate these aspects of the model and share their results.

Additionally, as indicated in our discussion of issues relating to potential misuse of the model, it remains unclear what the long-term dynamics are of detecting outputs from these models. We conducted [in-house automated ML-based detection research](https://github.com/openai/gpt-2-output-dataset/tree/master/detector) using simple classifiers, zero shot, and fine-tuning methods. Our fine-tuned detector model reached accuracy levels of approximately 95%. However, no one detection method is a panacea; automated ML-based detection, human detection, human-machine teaming, and metadata-based detection are all methods that can be combined for more confident classification. Developing better approaches to detection today will give us greater intuitions when thinking about future models and could help us understand ahead of time if detection methods will eventually become ineffective.
"""})

print('Name:', model.name)
print('Version:', model.version)
```

- Define helper function retrieve tags

```
def get_tag(tagname):
    text = ''
    try:
        text = tags[tagname]
    except:
        print('Missing tag ' + tagname)
    finally:
        return text

    return text

def get_datasheet(tags):
    
    title = get_tag('title')
    description = get_tag('datasheet_description')
    details = get_tag('details')
    date = get_tag('date')
    modeltype = get_tag('type')
    version = get_tag('version')
    helpresources = get_tag('help')
    usecase_primary = get_tag('usecase_primary')
    usecase_secondary = get_tag('usecase_secondary')
    usecase_outofscope = get_tag('usecase_outofscope')
    dataset_description = get_tag('dataset_description')
    motivation = get_tag('motivation')
    caveats = get_tag('caveats')

    datasheet = ''
    datasheet+=markdown(f'# {title} \n {description} \n')
    datasheet+=markdown(f'## Model Details \n {details} \n')
    datasheet+=markdown(f'### Model date \n {date} \n')
    datasheet+=markdown(f'### Model type \n {modeltype} \n')
    datasheet+=markdown(f'### Model version \n {version} \n')
    datasheet+=markdown(f'### Where to send questions or comments about the model \n Please send questions or concerns using [{helpresources}]({helpresources}) \n')
    datasheet+=markdown('## Intended Uses:\n')
    datasheet+=markdown(f'### Primary use case \n {usecase_primary} \n')
    datasheet+=markdown(f'### Secondary use case \n {usecase_secondary} \n')
    datasheet+=markdown(f'### Out of scope \n {usecase_outofscope} \n')
    datasheet+=markdown('## Evaluation Data:\n')
    datasheet+=markdown(f'### Datasets \n {dataset_description} \n')
    datasheet+=markdown(f'### Motivation \n {motivation} \n')
    datasheet+=markdown(f'### Caveats \n {caveats} \n')

    return datasheet
```

- Retrieve the registered model 

```
model = ws.models['gpt-2']
```

- Finally retrieve tags and see the model output

```
from IPython.core.display import display,Markdown

tags = model.tags
display(Markdown(get_datasheet(tags)))
```

inspired by:
https://github.com/microsoft/MLOps/blob/master/pytorch_with_datasheet/model_with_datasheet.ipynb