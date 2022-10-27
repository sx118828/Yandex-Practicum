# Обучение модели (BERT) классификации комментариев

Интернет-магазин «Y» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в Y-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию.

Необходимо обучить модель классифицировать комментарии на позитивные и негативные. В распоряжении набор данных с разметкой о токсичности.
Требование бизнеса - модель со значением метрики качества F1 не меньше 0.75.

## Загрузка данных, библиотек


```python
!pip install transformers
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: transformers in /usr/local/lib/python3.7/dist-packages (4.22.2)
    Requirement already satisfied: tokenizers!=0.11.3,<0.13,>=0.11.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.12.1)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (1.21.6)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.7/dist-packages (from transformers) (2022.6.2)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.7/dist-packages (from transformers) (21.3)
    Requirement already satisfied: importlib-metadata in /usr/local/lib/python3.7/dist-packages (from transformers) (4.12.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.7/dist-packages (from transformers) (3.8.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.9.0 in /usr/local/lib/python3.7/dist-packages (from transformers) (0.10.0)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.7/dist-packages (from transformers) (4.64.1)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.7/dist-packages (from transformers) (6.0)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.7/dist-packages (from huggingface-hub<1.0,>=0.9.0->transformers) (4.1.1)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=20.0->transformers) (3.0.9)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata->transformers) (3.8.1)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2022.6.15)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->transformers) (2.10)
    


```python
!pip install catboost
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: catboost in /usr/local/lib/python3.7/dist-packages (1.1)
    Requirement already satisfied: plotly in /usr/local/lib/python3.7/dist-packages (from catboost) (5.5.0)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.7/dist-packages (from catboost) (0.10.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from catboost) (1.7.3)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from catboost) (3.2.2)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.21.6)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from catboost) (1.15.0)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.7/dist-packages (from catboost) (1.3.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.24.0->catboost) (2022.2.1)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (0.11.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (1.4.4)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->catboost) (3.0.9)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from kiwisolver>=1.0.1->matplotlib->catboost) (4.1.1)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from plotly->catboost) (8.0.1)
    


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch
from torch.nn.utils.rnn import pad_sequence
import transformers as ppb
import os
import warnings
warnings.filterwarnings('ignore')

import re
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
nltk.download("stopwords")
from nltk.corpus import stopwords as nltk_stopwords

from tqdm import tqdm
tqdm.pandas()
from tqdm import notebook 
```

    [nltk_data] Downloading package omw-1.4 to /root/nltk_data...
    [nltk_data]   Package omw-1.4 is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
# просмотр, где находится каталог с файлами на COLAB
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
# получение доступа к каталогу и уточнение названия папок
import os
os.listdir('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 13 «Викишоп»/ДАННЫЕ')
```




    ['toxic_comments.csv', 'cased_L-12_H-768_A-12', 'rndlr96_EnBERT']




```python
# загрузка данных
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 13 «Викишоп»/ДАННЫЕ/toxic_comments.csv')
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 159571 entries, 0 to 159570
    Data columns (total 2 columns):
     #   Column  Non-Null Count   Dtype 
    ---  ------  --------------   ----- 
     0   text    159571 non-null  object
     1   toxic   159571 non-null  int64 
    dtypes: int64(1), object(1)
    memory usage: 2.4+ MB
    


```python
data.duplicated().sum() # подсчёт явных дубликатов
```




    0



### Вывод

**В результате загрузки данных, установлено:**

1.	DataFrame содержит 159571 строк и 2 столбца.
2.	В столбце «text» данные типа object, пропуски отсутствуют.
3.	В столбце «toxic» данные типа int64, пропуски отсутствуют.
4.	Явные дубликаты отсутствуют.


## Предобработка данных

#### Очистка данных


```python
data
```





  <div id="df-8d0a909d-7f8e-4fa3-9705-cc9d280e1d7b">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Explanation\nWhy the edits made under my usern...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>D'aww! He matches this background colour I'm s...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hey man, I'm really not trying to edit war. It...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"\nMore\nI can't make any real suggestions on ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>You, sir, are my hero. Any chance you remember...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>159566</th>
      <td>":::::And for the second time of asking, when ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>159567</th>
      <td>You should be ashamed of yourself \n\nThat is ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>159568</th>
      <td>Spitzer \n\nUmm, theres no actual article for ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>159569</th>
      <td>And it looks like it was actually you who put ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>159570</th>
      <td>"\nAnd ... I really don't think you understand...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>159571 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-8d0a909d-7f8e-4fa3-9705-cc9d280e1d7b')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-8d0a909d-7f8e-4fa3-9705-cc9d280e1d7b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-8d0a909d-7f8e-4fa3-9705-cc9d280e1d7b');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# функция лемматизации
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemm_list = nltk.word_tokenize(text)
    lemm_text = " ".join([lemmatizer.lemmatize(l) for l in lemm_list])     
    return lemm_text
```


```python
# функция очистки
def clear_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return " ".join(text.split())
```


```python
data['clear_text'] = data['text'].progress_apply(clear_text)
data['clear_text'] = data['clear_text'].progress_apply(lemmatize)
```

    100%|██████████| 159571/159571 [00:09<00:00, 16076.77it/s]
    100%|██████████| 159571/159571 [01:51<00:00, 1435.24it/s]
    

### light выборка

*Так как процесс моделирования на всем датафреме достаточно ресурсозатратный, выберем случайным образом 500 строк и обучим на них модель*


```python
df = data.sample(500, random_state=23031998).reset_index(drop=True)
```

### Проверим баланс классов


```python
print("Соотношение класса '0' и '1' соответственно:",
      round(df[df["toxic"] == 0].shape[0] / (df[df["toxic"] == 0].shape[0] + df[df["toxic"] == 1].shape[0]), 2), ":",
      round(df[df["toxic"] == 1].shape[0] / (df[df["toxic"] == 0].shape[0] + df[df["toxic"] == 1].shape[0]), 2))
```

    Соотношение класса '0' и '1' соответственно: 0.88 : 0.12
    

### Вывод

**В результате предобработки:**

1. Произведена очистка данных и лемматизация (для BERT можно и не делать лемматизацию, она и так хорошо справляется).
2. Выбрано случайным образом 500 строк, ввиду ресурсоёмкости обработки всего массива данных.
3. Установлено, что соотношение класса '0' и '1' соответственно: 0.88 : 0.12, данный факт указывает на необходимость балансировки, однако, для начала необходимо подготовить модели.

#### Загрузка предобученной BERT


```python
#model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased') #данная модель работает с длинной токена 512

```


```python
# Загрузка предобученной модели/токенизатора 
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
```

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel: ['cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 'cls.seq_relationship.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    


```python
tokenized = df['clear_text'].progress_apply(
    lambda x: tokenizer.encode(x, max_length=512, truncation=True, add_special_tokens=True)) #обрезаем под нужную длину токена

padded = pad_sequence([torch.as_tensor(seq) for seq in tokenized], batch_first=True) #выравниваем длину нулями  

attention_mask = padded > 0
attention_mask = attention_mask.type(torch.LongTensor)
```

    100%|██████████| 500/500 [00:01<00:00, 451.46it/s]
    

#### features & target


```python
%%time
batch_size = 100
embeddings = []
for i in notebook.tqdm(range(padded.shape[0] // batch_size)):
        batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)]) 
        attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])
        
        with torch.no_grad():
            batch_embeddings = model(batch, attention_mask=attention_mask_batch)
        
        embeddings.append(batch_embeddings[0][:,0,:].numpy())

features = np.concatenate(embeddings) 
```


      0%|          | 0/5 [00:00<?, ?it/s]


    CPU times: user 14min 59s, sys: 10.3 s, total: 15min 10s
    Wall time: 15min 12s
    


```python
features_train, features_test, target_train, target_test = train_test_split(features, df['toxic'], test_size=0.25)
```

### Вывод

**В результате подготовки модели, а также features & target:**

1.  Загружена предобученная модель `BertModel`.
2.  Выполнена токенизация и кодирование строк.
3.  Выполнена подрезка длинны токенов (не должно превышать 512 (обусловлено особенностями используемой модели BERT) 
4.  Выборка разделена на тренировочную и тестовую (75:25).


## Моделирование

### Поиск гиперпараметров

####  LogisticRegression


```python
# функция поиска best_score и параметров модели LogisticRegression
def LogisticRegression_model(features_train, target_train):
  model = LogisticRegression()
  parametrs = { 'C': range(10, 30, 1),
               'class_weight':['balanced', None] }
  search = HalvingGridSearchCV(model, parametrs, cv=5, scoring='f1')
  search.fit(features_train, target_train)
  best_model_LogisticRegression = search.best_estimator_
  best_score_model_LogisticRegression = round(search.best_score_, 3)
  
  return best_model_LogisticRegression, best_score_model_LogisticRegression
```


```python
best_model_LogisticRegression, best_score_model_LogisticRegression = LogisticRegression_model(features_train, target_train)
```

#### CatBoostClassifier


```python
# функция поиска best_score и параметров модели CatBoostClassifier
def CatBoostClassifier_model(features_train, target_train):
  model = CatBoostClassifier()
  parametrs = { 'depth': range (1, 3, 1),
              'n_estimators': range (1, 10, 1) }
  search = HalvingGridSearchCV(model, parametrs, cv=5, scoring='f1')
  search.fit(features_train, target_train)
  best_model_CatBoostClassifier = search.best_estimator_
  best_score_model_CatBoostClassifier = round(search.best_score_, 3)
  
  return best_model_CatBoostClassifier, best_score_model_CatBoostClassifier
```


```python
best_model_CatBoostClassifier, best_score_model_CatBoostClassifier = CatBoostClassifier_model(features_train, target_train)
```

    Learning rate set to 0.5
    0:	learn: 0.4569468	total: 6.56ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.2084242	total: 3.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3874567	total: 3.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3874567	total: 3.05ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4505279	total: 3.04ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4569468	total: 3.14ms	remaining: 3.14ms
    1:	learn: 0.3628890	total: 6.67ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.2084242	total: 3.05ms	remaining: 3.05ms
    1:	learn: 0.0683496	total: 6.55ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3874567	total: 3.01ms	remaining: 3.01ms
    1:	learn: 0.1749870	total: 6.35ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3874567	total: 3.05ms	remaining: 3.05ms
    1:	learn: 0.1723719	total: 6.42ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4505279	total: 3.1ms	remaining: 3.1ms
    1:	learn: 0.3741724	total: 6.48ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.4603198	total: 3.09ms	remaining: 6.18ms
    1:	learn: 0.3653077	total: 6.37ms	remaining: 3.19ms
    2:	learn: 0.2854861	total: 9.6ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.2138181	total: 2.98ms	remaining: 5.96ms
    1:	learn: 0.0708823	total: 6.17ms	remaining: 3.08ms
    2:	learn: 0.0362630	total: 9.42ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3917440	total: 3.01ms	remaining: 6.03ms
    1:	learn: 0.1779319	total: 6.78ms	remaining: 3.39ms
    2:	learn: 0.1151659	total: 9.77ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3917440	total: 3ms	remaining: 5.99ms
    1:	learn: 0.1754112	total: 6.27ms	remaining: 3.14ms
    2:	learn: 0.0786091	total: 9.36ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.4538956	total: 3.15ms	remaining: 6.3ms
    1:	learn: 0.3762900	total: 6.46ms	remaining: 3.23ms
    2:	learn: 0.2487205	total: 9.76ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4962532	total: 4.16ms	remaining: 12.5ms
    1:	learn: 0.3946258	total: 9.44ms	remaining: 9.44ms
    2:	learn: 0.3122784	total: 12.6ms	remaining: 4.19ms
    3:	learn: 0.2223073	total: 15.7ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.2747486	total: 2.94ms	remaining: 8.81ms
    1:	learn: 0.1645993	total: 6.11ms	remaining: 6.11ms
    2:	learn: 0.1111545	total: 9.22ms	remaining: 3.07ms
    3:	learn: 0.0628143	total: 12.3ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4379146	total: 3.04ms	remaining: 9.13ms
    1:	learn: 0.2147906	total: 6.26ms	remaining: 6.26ms
    2:	learn: 0.1386589	total: 9.38ms	remaining: 3.13ms
    3:	learn: 0.0710760	total: 12.7ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4379146	total: 2.97ms	remaining: 8.9ms
    1:	learn: 0.2993327	total: 6.19ms	remaining: 6.19ms
    2:	learn: 0.1650608	total: 9.29ms	remaining: 3.1ms
    3:	learn: 0.0911103	total: 12.3ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4902804	total: 3.06ms	remaining: 9.18ms
    1:	learn: 0.4029602	total: 6.41ms	remaining: 6.41ms
    2:	learn: 0.2826310	total: 9.52ms	remaining: 3.17ms
    3:	learn: 0.2276905	total: 12.6ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.5231306	total: 2.98ms	remaining: 11.9ms
    1:	learn: 0.4212065	total: 6.09ms	remaining: 9.14ms
    2:	learn: 0.3371484	total: 9.2ms	remaining: 6.13ms
    3:	learn: 0.2508992	total: 13ms	remaining: 3.25ms
    4:	learn: 0.2122984	total: 19.3ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.3242385	total: 4.02ms	remaining: 16.1ms
    1:	learn: 0.1991106	total: 7.52ms	remaining: 11.3ms
    2:	learn: 0.1394811	total: 10.8ms	remaining: 7.17ms
    3:	learn: 0.0814273	total: 13.9ms	remaining: 3.47ms
    4:	learn: 0.0535681	total: 17.1ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4727741	total: 3.13ms	remaining: 12.5ms
    1:	learn: 0.3726966	total: 6.63ms	remaining: 9.95ms
    2:	learn: 0.2188823	total: 9.92ms	remaining: 6.61ms
    3:	learn: 0.1160486	total: 13.2ms	remaining: 3.29ms
    4:	learn: 0.0948072	total: 16.2ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4727741	total: 3.06ms	remaining: 12.3ms
    1:	learn: 0.3295843	total: 6.7ms	remaining: 10.1ms
    2:	learn: 0.1899742	total: 9.96ms	remaining: 6.64ms
    3:	learn: 0.1134281	total: 13.2ms	remaining: 3.29ms
    4:	learn: 0.0889150	total: 16.3ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.5178522	total: 3.03ms	remaining: 12.1ms
    1:	learn: 0.4279828	total: 6.23ms	remaining: 9.34ms
    2:	learn: 0.3121781	total: 9.29ms	remaining: 6.19ms
    3:	learn: 0.2563685	total: 12.4ms	remaining: 3.11ms
    4:	learn: 0.2323731	total: 15.6ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.5436487	total: 3.08ms	remaining: 15.4ms
    1:	learn: 0.4443652	total: 6.44ms	remaining: 12.9ms
    2:	learn: 0.3599257	total: 9.64ms	remaining: 9.64ms
    3:	learn: 0.2757293	total: 12.7ms	remaining: 6.37ms
    4:	learn: 0.2383005	total: 15.7ms	remaining: 3.14ms
    5:	learn: 0.2217534	total: 18.7ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.3640139	total: 3.08ms	remaining: 15.4ms
    1:	learn: 0.2442344	total: 6.43ms	remaining: 12.9ms
    2:	learn: 0.1714816	total: 9.59ms	remaining: 9.59ms
    3:	learn: 0.1029811	total: 19.6ms	remaining: 9.8ms
    4:	learn: 0.0686470	total: 24.7ms	remaining: 4.94ms
    5:	learn: 0.0492572	total: 27.9ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4994374	total: 3.05ms	remaining: 15.3ms
    1:	learn: 0.3974858	total: 6.39ms	remaining: 12.8ms
    2:	learn: 0.2443450	total: 9.51ms	remaining: 9.51ms
    3:	learn: 0.1388271	total: 12.7ms	remaining: 6.37ms
    4:	learn: 0.1137375	total: 16ms	remaining: 3.19ms
    5:	learn: 0.0909902	total: 19.6ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4994374	total: 5.92ms	remaining: 29.6ms
    1:	learn: 0.3573611	total: 11.3ms	remaining: 22.7ms
    2:	learn: 0.1864669	total: 14.6ms	remaining: 14.6ms
    3:	learn: 0.1183866	total: 17.9ms	remaining: 8.94ms
    4:	learn: 0.0947373	total: 20.9ms	remaining: 4.17ms
    5:	learn: 0.0882917	total: 23.9ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.5389860	total: 3.07ms	remaining: 15.3ms
    1:	learn: 0.4501073	total: 9.78ms	remaining: 19.6ms
    2:	learn: 0.3382832	total: 13.1ms	remaining: 13.1ms
    3:	learn: 0.2814116	total: 16.3ms	remaining: 8.13ms
    4:	learn: 0.2556721	total: 19.4ms	remaining: 3.88ms
    5:	learn: 0.2362016	total: 22.7ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5597280	total: 3.02ms	remaining: 18.1ms
    1:	learn: 0.4643113	total: 6.3ms	remaining: 15.8ms
    2:	learn: 0.3805743	total: 9.5ms	remaining: 12.7ms
    3:	learn: 0.2979051	total: 12.7ms	remaining: 9.49ms
    4:	learn: 0.2608281	total: 15.7ms	remaining: 6.29ms
    5:	learn: 0.2434068	total: 21.9ms	remaining: 3.65ms
    6:	learn: 0.2292882	total: 26.5ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.3962943	total: 2.94ms	remaining: 17.6ms
    1:	learn: 0.2739173	total: 6.14ms	remaining: 15.3ms
    2:	learn: 0.1938642	total: 9.2ms	remaining: 12.3ms
    3:	learn: 0.1602822	total: 12.3ms	remaining: 9.23ms
    4:	learn: 0.1055505	total: 15.6ms	remaining: 6.25ms
    5:	learn: 0.0747162	total: 18.7ms	remaining: 3.12ms
    6:	learn: 0.0557589	total: 22.2ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5203317	total: 5.61ms	remaining: 33.6ms
    1:	learn: 0.4195339	total: 9.44ms	remaining: 23.6ms
    2:	learn: 0.2682784	total: 12.4ms	remaining: 16.6ms
    3:	learn: 0.1605943	total: 17.2ms	remaining: 12.9ms
    4:	learn: 0.1318167	total: 20.9ms	remaining: 8.36ms
    5:	learn: 0.1072469	total: 23.8ms	remaining: 3.97ms
    6:	learn: 0.0996706	total: 28.6ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5203317	total: 3.05ms	remaining: 18.3ms
    1:	learn: 0.3820794	total: 6.62ms	remaining: 16.6ms
    2:	learn: 0.2132670	total: 9.79ms	remaining: 13ms
    3:	learn: 0.1399161	total: 12.8ms	remaining: 9.63ms
    4:	learn: 0.1129978	total: 15.9ms	remaining: 6.38ms
    5:	learn: 0.1050538	total: 19ms	remaining: 3.17ms
    6:	learn: 0.0896133	total: 22.3ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5555657	total: 3.59ms	remaining: 21.5ms
    1:	learn: 0.4693151	total: 9.63ms	remaining: 24.1ms
    2:	learn: 0.3614311	total: 12.7ms	remaining: 17ms
    3:	learn: 0.3037778	total: 16.1ms	remaining: 12.1ms
    4:	learn: 0.2763206	total: 19.3ms	remaining: 7.72ms
    5:	learn: 0.2562614	total: 22.5ms	remaining: 3.75ms
    6:	learn: 0.2291325	total: 25.6ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5726368	total: 2.96ms	remaining: 20.7ms
    1:	learn: 0.4814901	total: 6.16ms	remaining: 18.5ms
    2:	learn: 0.3991793	total: 9.18ms	remaining: 15.3ms
    3:	learn: 0.3400272	total: 12.2ms	remaining: 12.2ms
    4:	learn: 0.3089049	total: 15.6ms	remaining: 9.36ms
    5:	learn: 0.2808315	total: 18.6ms	remaining: 6.19ms
    6:	learn: 0.2608978	total: 21.6ms	remaining: 3.09ms
    7:	learn: 0.2473054	total: 26.9ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.4228584	total: 3.09ms	remaining: 21.6ms
    1:	learn: 0.3006179	total: 6.39ms	remaining: 19.2ms
    2:	learn: 0.2156194	total: 9.59ms	remaining: 16ms
    3:	learn: 0.1771080	total: 12.7ms	remaining: 12.7ms
    4:	learn: 0.1189975	total: 15.9ms	remaining: 9.52ms
    5:	learn: 0.0852422	total: 19.1ms	remaining: 6.37ms
    6:	learn: 0.0641482	total: 22.3ms	remaining: 3.19ms
    7:	learn: 0.0501522	total: 25.6ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5370997	total: 2.95ms	remaining: 20.7ms
    1:	learn: 0.4389267	total: 6.19ms	remaining: 18.6ms
    2:	learn: 0.2905232	total: 9.25ms	remaining: 15.4ms
    3:	learn: 0.1813812	total: 12.4ms	remaining: 12.4ms
    4:	learn: 0.1492143	total: 15.4ms	remaining: 9.22ms
    5:	learn: 0.1227642	total: 18.5ms	remaining: 6.17ms
    6:	learn: 0.1141770	total: 21.6ms	remaining: 3.09ms
    7:	learn: 0.0992819	total: 24.8ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5370997	total: 3.13ms	remaining: 21.9ms
    1:	learn: 0.4038575	total: 6.31ms	remaining: 18.9ms
    2:	learn: 0.3303715	total: 9.36ms	remaining: 15.6ms
    3:	learn: 0.2187280	total: 12.3ms	remaining: 12.3ms
    4:	learn: 0.1748352	total: 15.3ms	remaining: 9.19ms
    5:	learn: 0.1338566	total: 18.4ms	remaining: 6.13ms
    6:	learn: 0.1151187	total: 21.5ms	remaining: 3.07ms
    7:	learn: 0.0985827	total: 24.6ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5688831	total: 3.01ms	remaining: 21.1ms
    1:	learn: 0.4859259	total: 6.21ms	remaining: 18.6ms
    2:	learn: 0.3819978	total: 9.15ms	remaining: 15.2ms
    3:	learn: 0.3239593	total: 12.2ms	remaining: 12.2ms
    4:	learn: 0.2949718	total: 15.2ms	remaining: 9.11ms
    5:	learn: 0.2741768	total: 18.3ms	remaining: 6.11ms
    6:	learn: 0.2477960	total: 21.5ms	remaining: 3.06ms
    7:	learn: 0.2241387	total: 24.6ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5832173	total: 3.15ms	remaining: 25.2ms
    1:	learn: 0.4963530	total: 5.97ms	remaining: 20.9ms
    2:	learn: 0.4159215	total: 8.71ms	remaining: 17.4ms
    3:	learn: 0.3886209	total: 11.4ms	remaining: 14.3ms
    4:	learn: 0.3531207	total: 14.1ms	remaining: 11.3ms
    5:	learn: 0.2890384	total: 16.8ms	remaining: 8.42ms
    6:	learn: 0.2692853	total: 19.5ms	remaining: 5.57ms
    7:	learn: 0.2580702	total: 22.2ms	remaining: 2.77ms
    8:	learn: 0.2411497	total: 24.8ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.4450434	total: 2.98ms	remaining: 23.8ms
    1:	learn: 0.3245093	total: 6.16ms	remaining: 21.5ms
    2:	learn: 0.2363735	total: 9.25ms	remaining: 18.5ms
    3:	learn: 0.1938305	total: 12.3ms	remaining: 15.4ms
    4:	learn: 0.1326866	total: 15.6ms	remaining: 12.5ms
    5:	learn: 0.1145537	total: 18.6ms	remaining: 9.32ms
    6:	learn: 0.0855464	total: 21.7ms	remaining: 6.19ms
    7:	learn: 0.0663751	total: 24.6ms	remaining: 3.08ms
    8:	learn: 0.0530911	total: 27.5ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5508330	total: 3.01ms	remaining: 24.1ms
    1:	learn: 0.4559433	total: 6.26ms	remaining: 21.9ms
    2:	learn: 0.3110394	total: 11.8ms	remaining: 23.6ms
    3:	learn: 0.2726842	total: 15ms	remaining: 18.8ms
    4:	learn: 0.2419920	total: 18.3ms	remaining: 14.6ms
    5:	learn: 0.2076397	total: 21.6ms	remaining: 10.8ms
    6:	learn: 0.1921787	total: 24.8ms	remaining: 7.09ms
    7:	learn: 0.1612411	total: 27.8ms	remaining: 3.48ms
    8:	learn: 0.1354254	total: 31ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5508330	total: 3.07ms	remaining: 24.5ms
    1:	learn: 0.4230176	total: 6.4ms	remaining: 22.4ms
    2:	learn: 0.3489812	total: 9.66ms	remaining: 19.3ms
    3:	learn: 0.2378161	total: 13.1ms	remaining: 16.4ms
    4:	learn: 0.1919168	total: 16.4ms	remaining: 13.1ms
    5:	learn: 0.1487532	total: 19.6ms	remaining: 9.79ms
    6:	learn: 0.1284093	total: 22.8ms	remaining: 6.5ms
    7:	learn: 0.1051409	total: 25.9ms	remaining: 3.24ms
    8:	learn: 0.0925910	total: 29ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5797963	total: 2.99ms	remaining: 23.9ms
    1:	learn: 0.5003417	total: 6.16ms	remaining: 21.6ms
    2:	learn: 0.4003164	total: 9.15ms	remaining: 18.3ms
    3:	learn: 0.3422710	total: 12.1ms	remaining: 15.1ms
    4:	learn: 0.3120140	total: 15.3ms	remaining: 12.2ms
    5:	learn: 0.2904595	total: 18.2ms	remaining: 9.12ms
    6:	learn: 0.2644750	total: 21.5ms	remaining: 6.13ms
    7:	learn: 0.2408787	total: 24.5ms	remaining: 3.06ms
    8:	learn: 0.2012123	total: 27.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4145603	total: 6.27ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.2213321	total: 5.81ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3536982	total: 5.81ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3914948	total: 5.93ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3656005	total: 5.88ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4145603	total: 6.1ms	remaining: 6.1ms
    1:	learn: 0.2276443	total: 12.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.2213321	total: 5.93ms	remaining: 5.93ms
    1:	learn: 0.0806064	total: 12ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3536982	total: 6.32ms	remaining: 6.32ms
    1:	learn: 0.1270551	total: 12.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3914948	total: 5.8ms	remaining: 5.8ms
    1:	learn: 0.1856898	total: 11.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3656005	total: 5.71ms	remaining: 5.71ms
    1:	learn: 0.2929181	total: 11.8ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.4185153	total: 5.71ms	remaining: 11.4ms
    1:	learn: 0.2323269	total: 11.7ms	remaining: 5.83ms
    2:	learn: 0.1272167	total: 17.4ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.2273655	total: 10.1ms	remaining: 20.1ms
    1:	learn: 0.0837716	total: 15.7ms	remaining: 7.86ms
    2:	learn: 0.0415977	total: 21.2ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3581509	total: 5.77ms	remaining: 11.5ms
    1:	learn: 0.1309507	total: 11.8ms	remaining: 5.88ms
    2:	learn: 0.0567862	total: 17.5ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3960091	total: 5.76ms	remaining: 11.5ms
    1:	learn: 0.1891207	total: 11.6ms	remaining: 5.79ms
    2:	learn: 0.0895863	total: 18.4ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3696184	total: 5.78ms	remaining: 11.6ms
    1:	learn: 0.2964565	total: 16.1ms	remaining: 8.05ms
    2:	learn: 0.1680259	total: 22.1ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4607204	total: 5.83ms	remaining: 17.5ms
    1:	learn: 0.2839701	total: 11.8ms	remaining: 11.8ms
    2:	learn: 0.1709336	total: 17.5ms	remaining: 5.84ms
    3:	learn: 0.1055410	total: 23.3ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.2931066	total: 5.79ms	remaining: 17.4ms
    1:	learn: 0.1716887	total: 11.7ms	remaining: 11.7ms
    2:	learn: 0.0808359	total: 19.2ms	remaining: 6.38ms
    3:	learn: 0.0532784	total: 25ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4064515	total: 9.75ms	remaining: 29.3ms
    1:	learn: 0.2442324	total: 16.6ms	remaining: 16.6ms
    2:	learn: 0.1453559	total: 22.5ms	remaining: 7.51ms
    3:	learn: 0.0852010	total: 28.3ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4435248	total: 6.08ms	remaining: 18.2ms
    1:	learn: 0.2630582	total: 12.1ms	remaining: 12.1ms
    2:	learn: 0.1367101	total: 18ms	remaining: 6.01ms
    3:	learn: 0.0990705	total: 24.2ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4139283	total: 5.72ms	remaining: 17.2ms
    1:	learn: 0.3363808	total: 11.5ms	remaining: 11.5ms
    2:	learn: 0.2455249	total: 17.7ms	remaining: 5.89ms
    3:	learn: 0.1836952	total: 23.5ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4923924	total: 6.37ms	remaining: 25.5ms
    1:	learn: 0.3258994	total: 12.5ms	remaining: 18.8ms
    2:	learn: 0.2090880	total: 18.6ms	remaining: 12.4ms
    3:	learn: 0.1356423	total: 24.7ms	remaining: 6.18ms
    4:	learn: 0.1109607	total: 30.6ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.3440856	total: 5.87ms	remaining: 23.5ms
    1:	learn: 0.2109945	total: 11.9ms	remaining: 17.8ms
    2:	learn: 0.1247106	total: 17.8ms	remaining: 11.9ms
    3:	learn: 0.0806878	total: 23.7ms	remaining: 5.92ms
    4:	learn: 0.0541632	total: 29.4ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4436120	total: 6.16ms	remaining: 24.6ms
    1:	learn: 0.2979072	total: 12.4ms	remaining: 18.5ms
    2:	learn: 0.1927245	total: 18.5ms	remaining: 12.3ms
    3:	learn: 0.1346533	total: 24.5ms	remaining: 6.12ms
    4:	learn: 0.0823942	total: 30.3ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4785395	total: 5.84ms	remaining: 23.4ms
    1:	learn: 0.2977538	total: 11.3ms	remaining: 17ms
    2:	learn: 0.1699997	total: 16.7ms	remaining: 11.1ms
    3:	learn: 0.1056551	total: 22.4ms	remaining: 5.59ms
    4:	learn: 0.0688156	total: 28.2ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4487786	total: 6.03ms	remaining: 24.1ms
    1:	learn: 0.4017782	total: 15.8ms	remaining: 23.7ms
    2:	learn: 0.3019150	total: 21.8ms	remaining: 14.5ms
    3:	learn: 0.2652510	total: 27.7ms	remaining: 6.93ms
    4:	learn: 0.2161850	total: 33.6ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.5165908	total: 7.41ms	remaining: 37ms
    1:	learn: 0.3603690	total: 13.6ms	remaining: 27.2ms
    2:	learn: 0.2883621	total: 19.8ms	remaining: 19.8ms
    3:	learn: 0.2300608	total: 25.8ms	remaining: 12.9ms
    4:	learn: 0.1950243	total: 31.8ms	remaining: 6.35ms
    5:	learn: 0.1687013	total: 37.7ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.3839489	total: 5.85ms	remaining: 29.3ms
    1:	learn: 0.2465990	total: 11.7ms	remaining: 23.4ms
    2:	learn: 0.1524446	total: 17.7ms	remaining: 17.7ms
    3:	learn: 0.1009285	total: 23.6ms	remaining: 11.8ms
    4:	learn: 0.0741083	total: 29.5ms	remaining: 5.89ms
    5:	learn: 0.0530647	total: 36.2ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4724996	total: 5.78ms	remaining: 28.9ms
    1:	learn: 0.3276923	total: 12ms	remaining: 23.9ms
    2:	learn: 0.2219370	total: 18ms	remaining: 18ms
    3:	learn: 0.1582286	total: 24ms	remaining: 12ms
    4:	learn: 0.1012542	total: 29.6ms	remaining: 5.92ms
    5:	learn: 0.0837083	total: 35.6ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.5050008	total: 5.88ms	remaining: 29.4ms
    1:	learn: 0.3288191	total: 11.9ms	remaining: 23.7ms
    2:	learn: 0.2007309	total: 18ms	remaining: 18ms
    3:	learn: 0.1282413	total: 24.3ms	remaining: 12.1ms
    4:	learn: 0.0862259	total: 30.6ms	remaining: 6.13ms
    5:	learn: 0.0719070	total: 41.4ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4762459	total: 5.9ms	remaining: 29.5ms
    1:	learn: 0.4268280	total: 12.8ms	remaining: 25.6ms
    2:	learn: 0.3295905	total: 18.8ms	remaining: 18.8ms
    3:	learn: 0.2921292	total: 24.7ms	remaining: 12.3ms
    4:	learn: 0.2423399	total: 30.5ms	remaining: 6.1ms
    5:	learn: 0.1760319	total: 36.1ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5355656	total: 5.76ms	remaining: 34.6ms
    1:	learn: 0.3890596	total: 11.7ms	remaining: 29.2ms
    2:	learn: 0.3155044	total: 17.8ms	remaining: 23.7ms
    3:	learn: 0.2469521	total: 23.7ms	remaining: 17.8ms
    4:	learn: 0.2302269	total: 29.6ms	remaining: 11.8ms
    5:	learn: 0.1977681	total: 35.3ms	remaining: 5.89ms
    6:	learn: 0.1510518	total: 41.2ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.4157140	total: 6.03ms	remaining: 36.2ms
    1:	learn: 0.2782638	total: 12.2ms	remaining: 30.5ms
    2:	learn: 0.1789929	total: 21.8ms	remaining: 29.1ms
    3:	learn: 0.1486479	total: 27.9ms	remaining: 20.9ms
    4:	learn: 0.1083568	total: 33.8ms	remaining: 13.5ms
    5:	learn: 0.0823707	total: 39.6ms	remaining: 6.59ms
    6:	learn: 0.0657621	total: 45.4ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.4954260	total: 5.82ms	remaining: 34.9ms
    1:	learn: 0.3541214	total: 12ms	remaining: 30.1ms
    2:	learn: 0.2398839	total: 21ms	remaining: 28ms
    3:	learn: 0.1871778	total: 26.9ms	remaining: 20.2ms
    4:	learn: 0.1243977	total: 32.7ms	remaining: 13.1ms
    5:	learn: 0.1045721	total: 38.5ms	remaining: 6.42ms
    6:	learn: 0.0772745	total: 44.3ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5255963	total: 5.82ms	remaining: 34.9ms
    1:	learn: 0.3561345	total: 16.3ms	remaining: 40.6ms
    2:	learn: 0.2288962	total: 22.8ms	remaining: 30.4ms
    3:	learn: 0.1520188	total: 28.8ms	remaining: 21.6ms
    4:	learn: 0.1046973	total: 34.7ms	remaining: 13.9ms
    5:	learn: 0.0875049	total: 40.6ms	remaining: 6.76ms
    6:	learn: 0.0738942	total: 46.4ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.4982490	total: 5.91ms	remaining: 35.4ms
    1:	learn: 0.4482278	total: 11.5ms	remaining: 28.8ms
    2:	learn: 0.3538941	total: 17.3ms	remaining: 23.1ms
    3:	learn: 0.3156313	total: 23.2ms	remaining: 17.4ms
    4:	learn: 0.2653114	total: 28.9ms	remaining: 11.6ms
    5:	learn: 0.2273383	total: 34.9ms	remaining: 5.82ms
    6:	learn: 0.1735214	total: 40.8ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5507989	total: 5.87ms	remaining: 41.1ms
    1:	learn: 0.4132314	total: 11.9ms	remaining: 35.7ms
    2:	learn: 0.3649260	total: 17.9ms	remaining: 29.8ms
    3:	learn: 0.2612781	total: 23.8ms	remaining: 23.8ms
    4:	learn: 0.2403561	total: 29.7ms	remaining: 17.8ms
    5:	learn: 0.2119421	total: 35.5ms	remaining: 11.8ms
    6:	learn: 0.1897901	total: 47.6ms	remaining: 6.81ms
    7:	learn: 0.1587300	total: 54.8ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.4415286	total: 5.87ms	remaining: 41.1ms
    1:	learn: 0.3062500	total: 12.1ms	remaining: 36.2ms
    2:	learn: 0.2341640	total: 18ms	remaining: 30ms
    3:	learn: 0.1943566	total: 24ms	remaining: 24ms
    4:	learn: 0.1427128	total: 30.1ms	remaining: 18.1ms
    5:	learn: 0.1192368	total: 36.4ms	remaining: 12.1ms
    6:	learn: 0.0951091	total: 42.7ms	remaining: 6.1ms
    7:	learn: 0.0734636	total: 48.8ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5139987	total: 6.51ms	remaining: 45.6ms
    1:	learn: 0.3774099	total: 13ms	remaining: 39ms
    2:	learn: 0.2632802	total: 19.4ms	remaining: 32.3ms
    3:	learn: 0.2079443	total: 25.9ms	remaining: 25.9ms
    4:	learn: 0.1423216	total: 32.1ms	remaining: 19.3ms
    5:	learn: 0.1203961	total: 39.3ms	remaining: 13.1ms
    6:	learn: 0.1081539	total: 45.5ms	remaining: 6.5ms
    7:	learn: 0.0928379	total: 55.9ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5420523	total: 5.87ms	remaining: 41.1ms
    1:	learn: 0.3800419	total: 12ms	remaining: 36.1ms
    2:	learn: 0.2545775	total: 18ms	remaining: 30.1ms
    3:	learn: 0.1761529	total: 24.2ms	remaining: 24.2ms
    4:	learn: 0.1247571	total: 30.2ms	remaining: 18.1ms
    5:	learn: 0.1052429	total: 36.2ms	remaining: 12.1ms
    6:	learn: 0.0895986	total: 42.2ms	remaining: 6.03ms
    7:	learn: 0.0812212	total: 48.1ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5161940	total: 5.89ms	remaining: 41.2ms
    1:	learn: 0.4665843	total: 11.8ms	remaining: 35.3ms
    2:	learn: 0.3741106	total: 17.8ms	remaining: 29.7ms
    3:	learn: 0.3429766	total: 23.8ms	remaining: 23.8ms
    4:	learn: 0.2954311	total: 29.6ms	remaining: 17.8ms
    5:	learn: 0.2261435	total: 35.4ms	remaining: 11.8ms
    6:	learn: 0.1772066	total: 41.2ms	remaining: 5.89ms
    7:	learn: 0.1605964	total: 47.6ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5632913	total: 5.79ms	remaining: 46.4ms
    1:	learn: 0.4338338	total: 11.8ms	remaining: 41.2ms
    2:	learn: 0.3851881	total: 17.7ms	remaining: 35.4ms
    3:	learn: 0.2821299	total: 23.4ms	remaining: 29.3ms
    4:	learn: 0.2555876	total: 29.3ms	remaining: 23.4ms
    5:	learn: 0.2249209	total: 35.1ms	remaining: 17.5ms
    6:	learn: 0.2025499	total: 40.8ms	remaining: 11.7ms
    7:	learn: 0.1717048	total: 46.6ms	remaining: 5.83ms
    8:	learn: 0.1624716	total: 52.7ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.4628822	total: 5.77ms	remaining: 46.2ms
    1:	learn: 0.3309745	total: 11.7ms	remaining: 41ms
    2:	learn: 0.2565126	total: 17.7ms	remaining: 35.5ms
    3:	learn: 0.2131346	total: 23.5ms	remaining: 29.4ms
    4:	learn: 0.1793083	total: 29.3ms	remaining: 23.5ms
    5:	learn: 0.1496209	total: 35.3ms	remaining: 17.6ms
    6:	learn: 0.1201478	total: 41.4ms	remaining: 11.8ms
    7:	learn: 0.0932780	total: 47.2ms	remaining: 5.9ms
    8:	learn: 0.0750618	total: 53ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5293301	total: 5.82ms	remaining: 46.6ms
    1:	learn: 0.3979340	total: 12.1ms	remaining: 42.2ms
    2:	learn: 0.2848509	total: 18ms	remaining: 35.9ms
    3:	learn: 0.2440460	total: 23.8ms	remaining: 29.7ms
    4:	learn: 0.1698818	total: 29.6ms	remaining: 23.6ms
    5:	learn: 0.1437999	total: 35.4ms	remaining: 17.7ms
    6:	learn: 0.1299054	total: 41.3ms	remaining: 11.8ms
    7:	learn: 0.1117848	total: 47.1ms	remaining: 5.89ms
    8:	learn: 0.0921456	total: 53ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5554922	total: 8.95ms	remaining: 71.6ms
    1:	learn: 0.4009821	total: 15.1ms	remaining: 52.8ms
    2:	learn: 0.2779353	total: 21ms	remaining: 42ms
    3:	learn: 0.1973472	total: 26.8ms	remaining: 33.5ms
    4:	learn: 0.1424602	total: 35.2ms	remaining: 28.1ms
    5:	learn: 0.1205555	total: 45.9ms	remaining: 22.9ms
    6:	learn: 0.1031642	total: 51.7ms	remaining: 14.8ms
    7:	learn: 0.0936097	total: 57.4ms	remaining: 7.18ms
    8:	learn: 0.0831237	total: 63.2ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5310740	total: 5.82ms	remaining: 46.6ms
    1:	learn: 0.4824359	total: 12ms	remaining: 41.9ms
    2:	learn: 0.4197369	total: 17.8ms	remaining: 35.5ms
    3:	learn: 0.3180737	total: 23.5ms	remaining: 29.4ms
    4:	learn: 0.2610552	total: 29.3ms	remaining: 23.4ms
    5:	learn: 0.2281262	total: 35.1ms	remaining: 17.5ms
    6:	learn: 0.1807298	total: 40.8ms	remaining: 11.7ms
    7:	learn: 0.1671329	total: 46.4ms	remaining: 5.8ms
    8:	learn: 0.1411257	total: 52.2ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4572973	total: 6.99ms	remaining: 27.9ms
    1:	learn: 0.3649160	total: 13.8ms	remaining: 20.7ms
    2:	learn: 0.3037804	total: 20.5ms	remaining: 13.7ms
    3:	learn: 0.2661878	total: 27.1ms	remaining: 6.78ms
    4:	learn: 0.2528618	total: 33.7ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4788930	total: 8.41ms	remaining: 33.6ms
    1:	learn: 0.3703874	total: 16.5ms	remaining: 24.7ms
    2:	learn: 0.3307753	total: 23ms	remaining: 15.3ms
    3:	learn: 0.3079404	total: 29.5ms	remaining: 7.37ms
    4:	learn: 0.2124077	total: 36.1ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4191187	total: 6.3ms	remaining: 25.2ms
    1:	learn: 0.2929854	total: 12.8ms	remaining: 19.2ms
    2:	learn: 0.2269853	total: 19.3ms	remaining: 12.9ms
    3:	learn: 0.1937426	total: 25.7ms	remaining: 6.44ms
    4:	learn: 0.1442763	total: 32.1ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4652960	total: 9.67ms	remaining: 38.7ms
    1:	learn: 0.3883540	total: 16ms	remaining: 24ms
    2:	learn: 0.3091590	total: 22.4ms	remaining: 15ms
    3:	learn: 0.2694894	total: 28.9ms	remaining: 7.21ms
    4:	learn: 0.2463002	total: 35.4ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4652960	total: 6.7ms	remaining: 26.8ms
    1:	learn: 0.3537303	total: 14ms	remaining: 21ms
    2:	learn: 0.2764798	total: 20.9ms	remaining: 14ms
    3:	learn: 0.2460669	total: 27.4ms	remaining: 6.84ms
    4:	learn: 0.2240755	total: 33.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4551230	total: 6.36ms	remaining: 19.1ms
    1:	learn: 0.3638781	total: 12.7ms	remaining: 12.7ms
    2:	learn: 0.3028758	total: 19.5ms	remaining: 6.51ms
    3:	learn: 0.2652109	total: 26.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4769118	total: 6.37ms	remaining: 19.1ms
    1:	learn: 0.3692657	total: 13.3ms	remaining: 13.3ms
    2:	learn: 0.3296280	total: 20.4ms	remaining: 6.8ms
    3:	learn: 0.3067923	total: 27.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4166642	total: 6.3ms	remaining: 18.9ms
    1:	learn: 0.2913765	total: 12.9ms	remaining: 12.9ms
    2:	learn: 0.2254923	total: 26.4ms	remaining: 8.79ms
    3:	learn: 0.1924198	total: 33.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4632021	total: 8.85ms	remaining: 26.5ms
    1:	learn: 0.3872863	total: 17.4ms	remaining: 17.4ms
    2:	learn: 0.3079698	total: 24.8ms	remaining: 8.26ms
    3:	learn: 0.2681125	total: 31.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4632021	total: 6.32ms	remaining: 19ms
    1:	learn: 0.3524117	total: 12.9ms	remaining: 12.9ms
    2:	learn: 0.2408288	total: 23.1ms	remaining: 7.71ms
    3:	learn: 0.2226553	total: 30.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4551230	total: 6.22ms	remaining: 12.4ms
    1:	learn: 0.3638781	total: 12.9ms	remaining: 6.44ms
    2:	learn: 0.3028758	total: 19.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4769118	total: 6.31ms	remaining: 12.6ms
    1:	learn: 0.3692657	total: 12.7ms	remaining: 6.37ms
    2:	learn: 0.3296280	total: 19.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4166642	total: 6.33ms	remaining: 12.7ms
    1:	learn: 0.2913765	total: 14ms	remaining: 6.98ms
    2:	learn: 0.2254923	total: 20.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4632021	total: 6.49ms	remaining: 13ms
    1:	learn: 0.3872863	total: 13.2ms	remaining: 6.61ms
    2:	learn: 0.3079698	total: 19.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4632021	total: 6.42ms	remaining: 12.8ms
    1:	learn: 0.3524117	total: 13.2ms	remaining: 6.61ms
    2:	learn: 0.2408288	total: 19.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4551230	total: 6.42ms	remaining: 6.42ms
    1:	learn: 0.3638781	total: 13.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4769118	total: 6.33ms	remaining: 6.33ms
    1:	learn: 0.3692657	total: 12.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4166642	total: 6.18ms	remaining: 6.18ms
    1:	learn: 0.2913765	total: 12.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4632021	total: 6.36ms	remaining: 6.36ms
    1:	learn: 0.3872863	total: 16ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4632021	total: 6.35ms	remaining: 6.35ms
    1:	learn: 0.3524117	total: 12.8ms	remaining: 0us
    Learning rate set to 0.319929
    0:	learn: 0.4942568	total: 13.7ms	remaining: 95.8ms
    1:	learn: 0.3961453	total: 27.2ms	remaining: 81.5ms
    2:	learn: 0.3133834	total: 44.8ms	remaining: 74.7ms
    3:	learn: 0.2659478	total: 58.9ms	remaining: 58.9ms
    4:	learn: 0.2093490	total: 72.3ms	remaining: 43.4ms
    5:	learn: 0.1776007	total: 85.5ms	remaining: 28.5ms
    6:	learn: 0.1628103	total: 98.9ms	remaining: 14.1ms
    7:	learn: 0.1347867	total: 112ms	remaining: 0us
    Learning rate set to 0.319929
    0:	learn: 0.5124564	total: 15.3ms	remaining: 107ms
    1:	learn: 0.4112799	total: 28.8ms	remaining: 86.4ms
    2:	learn: 0.3486828	total: 41.8ms	remaining: 69.7ms
    3:	learn: 0.3206133	total: 55.2ms	remaining: 55.2ms
    4:	learn: 0.2772643	total: 72ms	remaining: 43.2ms
    5:	learn: 0.2291494	total: 85.8ms	remaining: 28.6ms
    6:	learn: 0.1812256	total: 99.1ms	remaining: 14.2ms
    7:	learn: 0.1455116	total: 112ms	remaining: 0us
    Learning rate set to 0.319929
    0:	learn: 0.4791268	total: 13.1ms	remaining: 92ms
    1:	learn: 0.3748095	total: 28.2ms	remaining: 84.6ms
    2:	learn: 0.3045398	total: 43.2ms	remaining: 72.1ms
    3:	learn: 0.2502082	total: 56.4ms	remaining: 56.4ms
    4:	learn: 0.2185999	total: 69.4ms	remaining: 41.6ms
    5:	learn: 0.1714994	total: 82.3ms	remaining: 27.4ms
    6:	learn: 0.1571917	total: 99.9ms	remaining: 14.3ms
    7:	learn: 0.1444435	total: 113ms	remaining: 0us
    Learning rate set to 0.319929
    0:	learn: 0.5048091	total: 13ms	remaining: 91.1ms
    1:	learn: 0.4336716	total: 29ms	remaining: 87.1ms
    2:	learn: 0.3607132	total: 42.6ms	remaining: 71ms
    3:	learn: 0.3043263	total: 55.9ms	remaining: 55.9ms
    4:	learn: 0.2735113	total: 69.1ms	remaining: 41.4ms
    5:	learn: 0.2516464	total: 82.7ms	remaining: 27.6ms
    6:	learn: 0.2298243	total: 96.1ms	remaining: 13.7ms
    7:	learn: 0.2243116	total: 111ms	remaining: 0us
    Learning rate set to 0.319929
    0:	learn: 0.4876429	total: 13.3ms	remaining: 93.1ms
    1:	learn: 0.3463625	total: 27ms	remaining: 81.1ms
    2:	learn: 0.2875139	total: 40.6ms	remaining: 67.6ms
    3:	learn: 0.2473417	total: 54.1ms	remaining: 54.1ms
    4:	learn: 0.2101489	total: 67.7ms	remaining: 40.6ms
    5:	learn: 0.1727209	total: 81.2ms	remaining: 27.1ms
    6:	learn: 0.1511066	total: 94.3ms	remaining: 13.5ms
    7:	learn: 0.1323929	total: 108ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.5093090	total: 13.2ms	remaining: 105ms
    1:	learn: 0.4105210	total: 31.1ms	remaining: 109ms
    2:	learn: 0.3332185	total: 44.9ms	remaining: 89.9ms
    3:	learn: 0.2944360	total: 58.8ms	remaining: 73.5ms
    4:	learn: 0.2658368	total: 83.8ms	remaining: 67.1ms
    5:	learn: 0.2310414	total: 100ms	remaining: 50.1ms
    6:	learn: 0.2147810	total: 113ms	remaining: 32.4ms
    7:	learn: 0.1982479	total: 127ms	remaining: 15.9ms
    8:	learn: 0.1746593	total: 140ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.5263267	total: 13.2ms	remaining: 106ms
    1:	learn: 0.4259188	total: 31.7ms	remaining: 111ms
    2:	learn: 0.3658856	total: 44.7ms	remaining: 89.3ms
    3:	learn: 0.3359024	total: 57.5ms	remaining: 71.9ms
    4:	learn: 0.2907603	total: 72.3ms	remaining: 57.8ms
    5:	learn: 0.2498776	total: 85.5ms	remaining: 42.7ms
    6:	learn: 0.1988793	total: 98.6ms	remaining: 28.2ms
    7:	learn: 0.1837228	total: 112ms	remaining: 14ms
    8:	learn: 0.1617071	total: 126ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.4956436	total: 13.2ms	remaining: 106ms
    1:	learn: 0.3904549	total: 26.5ms	remaining: 92.6ms
    2:	learn: 0.3174007	total: 42ms	remaining: 84.1ms
    3:	learn: 0.2626486	total: 56.9ms	remaining: 71.2ms
    4:	learn: 0.2167005	total: 70.5ms	remaining: 56.4ms
    5:	learn: 0.1772986	total: 83.8ms	remaining: 41.9ms
    6:	learn: 0.1457087	total: 97.1ms	remaining: 27.7ms
    7:	learn: 0.1327998	total: 110ms	remaining: 13.8ms
    8:	learn: 0.1098720	total: 128ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.5192888	total: 14.3ms	remaining: 115ms
    1:	learn: 0.4467111	total: 28.1ms	remaining: 98.3ms
    2:	learn: 0.3719641	total: 41.6ms	remaining: 83.2ms
    3:	learn: 0.3101604	total: 60.1ms	remaining: 75.1ms
    4:	learn: 0.2553153	total: 74.1ms	remaining: 59.3ms
    5:	learn: 0.2337137	total: 87.4ms	remaining: 43.7ms
    6:	learn: 0.2181175	total: 101ms	remaining: 28.8ms
    7:	learn: 0.2093054	total: 114ms	remaining: 14.3ms
    8:	learn: 0.1756921	total: 127ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.5036803	total: 16.7ms	remaining: 134ms
    1:	learn: 0.3644769	total: 30.7ms	remaining: 108ms
    2:	learn: 0.3032032	total: 45.3ms	remaining: 90.5ms
    3:	learn: 0.2580042	total: 58.5ms	remaining: 73.2ms
    4:	learn: 0.2254279	total: 76.3ms	remaining: 61.1ms
    5:	learn: 0.2022112	total: 89.5ms	remaining: 44.8ms
    6:	learn: 0.1849669	total: 102ms	remaining: 29.2ms
    7:	learn: 0.1625101	total: 115ms	remaining: 14.4ms
    8:	learn: 0.1344136	total: 128ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4012101	total: 35.6ms	remaining: 249ms
    1:	learn: 0.3079351	total: 84.2ms	remaining: 253ms
    2:	learn: 0.2353240	total: 114ms	remaining: 190ms
    3:	learn: 0.2174010	total: 144ms	remaining: 144ms
    4:	learn: 0.1988359	total: 173ms	remaining: 104ms
    5:	learn: 0.1871771	total: 203ms	remaining: 67.7ms
    6:	learn: 0.1735613	total: 234ms	remaining: 33.5ms
    7:	learn: 0.1625868	total: 272ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4222055	total: 29.7ms	remaining: 208ms
    1:	learn: 0.3344231	total: 59.9ms	remaining: 180ms
    2:	learn: 0.2848499	total: 94.5ms	remaining: 158ms
    3:	learn: 0.2532086	total: 125ms	remaining: 125ms
    4:	learn: 0.2384652	total: 155ms	remaining: 92.8ms
    5:	learn: 0.2049532	total: 184ms	remaining: 61.5ms
    6:	learn: 0.1941427	total: 215ms	remaining: 30.7ms
    7:	learn: 0.1853970	total: 252ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3820591	total: 29.3ms	remaining: 205ms
    1:	learn: 0.2954358	total: 60.1ms	remaining: 180ms
    2:	learn: 0.2647730	total: 91.4ms	remaining: 152ms
    3:	learn: 0.2345630	total: 121ms	remaining: 121ms
    4:	learn: 0.2045293	total: 150ms	remaining: 90.2ms
    5:	learn: 0.1982194	total: 183ms	remaining: 61.2ms
    6:	learn: 0.1810872	total: 218ms	remaining: 31.1ms
    7:	learn: 0.1632965	total: 247ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4171293	total: 29.4ms	remaining: 206ms
    1:	learn: 0.3309517	total: 64.7ms	remaining: 194ms
    2:	learn: 0.2722654	total: 94.5ms	remaining: 157ms
    3:	learn: 0.2558273	total: 124ms	remaining: 124ms
    4:	learn: 0.2247683	total: 154ms	remaining: 92.2ms
    5:	learn: 0.1877545	total: 184ms	remaining: 61.2ms
    6:	learn: 0.1789837	total: 214ms	remaining: 30.5ms
    7:	learn: 0.1672437	total: 244ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4258388	total: 29.3ms	remaining: 205ms
    1:	learn: 0.3181723	total: 58.8ms	remaining: 176ms
    2:	learn: 0.2732142	total: 89.2ms	remaining: 149ms
    3:	learn: 0.2465179	total: 122ms	remaining: 122ms
    4:	learn: 0.2257341	total: 152ms	remaining: 91.1ms
    5:	learn: 0.2166207	total: 182ms	remaining: 60.5ms
    6:	learn: 0.2094720	total: 216ms	remaining: 30.9ms
    7:	learn: 0.1988651	total: 246ms	remaining: 0us
    Learning rate set to 0.459735
    0:	learn: 0.4141608	total: 32.8ms	remaining: 262ms
    1:	learn: 0.3154770	total: 65.9ms	remaining: 231ms
    2:	learn: 0.2411342	total: 96.1ms	remaining: 192ms
    3:	learn: 0.2237726	total: 128ms	remaining: 160ms
    4:	learn: 0.2052052	total: 157ms	remaining: 126ms
    5:	learn: 0.1939622	total: 191ms	remaining: 95.5ms
    6:	learn: 0.1799882	total: 228ms	remaining: 65.1ms
    7:	learn: 0.1688995	total: 258ms	remaining: 32.2ms
    8:	learn: 0.1643068	total: 294ms	remaining: 0us
    Learning rate set to 0.459735
    0:	learn: 0.4351188	total: 29.4ms	remaining: 235ms
    1:	learn: 0.3416712	total: 59ms	remaining: 207ms
    2:	learn: 0.2905814	total: 93.8ms	remaining: 188ms
    3:	learn: 0.2582229	total: 124ms	remaining: 155ms
    4:	learn: 0.2437029	total: 157ms	remaining: 125ms
    5:	learn: 0.2107746	total: 186ms	remaining: 93.2ms
    6:	learn: 0.1994555	total: 216ms	remaining: 61.8ms
    7:	learn: 0.1908645	total: 247ms	remaining: 30.8ms
    8:	learn: 0.1832616	total: 277ms	remaining: 0us
    Learning rate set to 0.459735
    0:	learn: 0.3957721	total: 30.6ms	remaining: 244ms
    1:	learn: 0.3033517	total: 66.9ms	remaining: 234ms
    2:	learn: 0.2705573	total: 96.8ms	remaining: 194ms
    3:	learn: 0.2404947	total: 127ms	remaining: 159ms
    4:	learn: 0.2110737	total: 157ms	remaining: 125ms
    5:	learn: 0.2050246	total: 186ms	remaining: 93.1ms
    6:	learn: 0.1898950	total: 223ms	remaining: 63.8ms
    7:	learn: 0.1718984	total: 253ms	remaining: 31.6ms
    8:	learn: 0.1551483	total: 283ms	remaining: 0us
    Learning rate set to 0.459735
    0:	learn: 0.4302305	total: 29.1ms	remaining: 233ms
    1:	learn: 0.3382965	total: 60.6ms	remaining: 212ms
    2:	learn: 0.2767555	total: 90.2ms	remaining: 180ms
    3:	learn: 0.2602752	total: 120ms	remaining: 150ms
    4:	learn: 0.2303702	total: 150ms	remaining: 120ms
    5:	learn: 0.2113301	total: 180ms	remaining: 89.8ms
    6:	learn: 0.2048883	total: 214ms	remaining: 61.2ms
    7:	learn: 0.1984621	total: 245ms	remaining: 30.6ms
    8:	learn: 0.1734960	total: 274ms	remaining: 0us
    Learning rate set to 0.459735
    0:	learn: 0.4386810	total: 34.2ms	remaining: 274ms
    1:	learn: 0.3223126	total: 64.1ms	remaining: 224ms
    2:	learn: 0.2783500	total: 93.6ms	remaining: 187ms
    3:	learn: 0.2528118	total: 123ms	remaining: 153ms
    4:	learn: 0.2338401	total: 152ms	remaining: 122ms
    5:	learn: 0.2257736	total: 182ms	remaining: 90.9ms
    6:	learn: 0.2174754	total: 212ms	remaining: 60.5ms
    7:	learn: 0.2030319	total: 245ms	remaining: 30.6ms
    8:	learn: 0.1950318	total: 279ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4160051	total: 33.4ms	remaining: 234ms
    1:	learn: 0.2879505	total: 63.4ms	remaining: 190ms
    2:	learn: 0.2618753	total: 92.9ms	remaining: 155ms
    3:	learn: 0.2492535	total: 123ms	remaining: 123ms
    4:	learn: 0.2229675	total: 152ms	remaining: 91.4ms
    5:	learn: 0.1896732	total: 196ms	remaining: 65.4ms
    6:	learn: 0.1823444	total: 228ms	remaining: 32.5ms
    7:	learn: 0.1702512	total: 261ms	remaining: 0us
    

#### RandomForestClassifier


```python
# функция поиска best_score и параметров модели RandomForestClassifier
def RandomForestClassifier_model(features_train, target_train):
  model = RandomForestClassifier()
  parametrs = { 'max_depth': range (6, 12, 1),
              'n_estimators': range (25, 30, 1) }
  search = HalvingGridSearchCV(model, parametrs, cv=5, scoring='f1')
  search.fit(features_train, target_train)
  best_model_RandomForestClassifier = search.best_estimator_
  best_score_model_RandomForestClassifier = round(search.best_score_, 3)

  return best_model_RandomForestClassifier, best_score_model_RandomForestClassifier
```


```python
best_model_RandomForestClassifier, best_score_model_RandomForestClassifier = RandomForestClassifier_model(features_train, target_train)
```

#### LGBMClassifier


```python
# функция поиска best_score и параметров модели LGBMClassifier
def LGBMClassifier_model(features_train, target_train):
  model = LGBMClassifier()
  parametrs = { 'max_depth': range (1, 5, 1),
              'n_estimators': range (1, 10, 1) }
  search = HalvingGridSearchCV(model, parametrs, cv=5, scoring='f1')
  search.fit(features_train, target_train)
  best_model_LGBMClassifier = search.best_estimator_
  best_score_model_LGBMClassifier = round(search.best_score_, 3)
  
  return best_model_LGBMClassifier, best_score_model_LGBMClassifier
```


```python
best_model_LGBMClassifier, best_score_model_LGBMClassifier = LGBMClassifier_model(features_train, target_train)
```

### Рейтинг моделей по метрике качества (F1)


```python
list_model = [best_model_LogisticRegression,
              best_model_CatBoostClassifier,
              best_model_RandomForestClassifier,
              best_model_LGBMClassifier]
              
list_score = [best_score_model_LogisticRegression,
              best_score_model_CatBoostClassifier,
              best_score_model_RandomForestClassifier,
              best_score_model_LGBMClassifier]
```


```python
intermediate_dictionary = {'Model':list_model, 'F1':list_score}
rating_model = pd.DataFrame(intermediate_dictionary)
rating = rating_model.sort_values(by='F1', ascending=False)
rating
```





  <div id="df-1fc44ba5-4cae-4644-a69d-769109b61354">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression(C=18, class_weight='balanced')</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;catboost.core.CatBoostClassifier object at 0x...</td>
      <td>0.303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(DecisionTreeClassifier(max_depth=6, max_featu...</td>
      <td>0.167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LGBMClassifier(max_depth=2, n_estimators=6)</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1fc44ba5-4cae-4644-a69d-769109b61354')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1fc44ba5-4cae-4644-a69d-769109b61354 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1fc44ba5-4cae-4644-a69d-769109b61354');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
print('Первое место занимает', rating.iloc[0,0])
print('Значении F1 равно', rating.iloc[0,1])
```

    Первое место занимает LogisticRegression(C=18, class_weight='balanced')
    Значении F1 равно 0.444
    


```python
best_model_on_data_default = rating.iloc[0,0]
```

### Вывод

**В результате обучения и подбора лучших параметров моделей default данных, установлено:**

1. Лучшая модель: LogisticRegression.
2. Параметры лучшей модели: C=18, class_weight='balanced'.
3. Качество модели (F1): 0.444.
4. Не высокое качество модели обусловлено дисбалансом классов.

## Поиск множителя балансировки, а также наиболее оптимальной модели


```python
target_train = pd.DataFrame(target_train)
features_train = pd.DataFrame(features_train)
```


```python
train = np.concatenate((features_train, target_train), axis=1)
train = pd.DataFrame(train)
```


```python
print("Соотношение класса '0' и '1' соответствено:",
      round(train[train.iloc[:,-1] == 0].shape[0] / (train[train.iloc[:,-1] == 0].shape[0] + train[train.iloc[:,-1] == 1].shape[0]), 2), ":",
      round(train[train.iloc[:,-1] == 1].shape[0] / (train[train.iloc[:,-1] == 0].shape[0] + train[train.iloc[:,-1] == 1].shape[0]), 2))
```

    Соотношение класса '0' и '1' соответствено: 0.88 : 0.12
    


```python
# функция увеличения выборки
def upsample(dataset, repeat):
  train_zeros = dataset[dataset.iloc[:,-1] == 0]
  train_ones = dataset[dataset.iloc[:,-1] == 1]
  train_upsampled = pd.concat([train_zeros] + [train_ones] * repeat)
  train_up = shuffle(
        train_upsampled, random_state=12345)
  return train_up
```


```python
# функция поиска наиболее оптимального модели, а также значения МНОЖИТЕЛЯ (хN), best_score и параметров модели
def search_best_xN_F1(dataset, n, m):
  best_xN = 0
  best_F1 = 0
  best_model = None
  best_target_up = None
  best_features_up = None
  for j in range(1, 5):
    train_up = upsample(dataset, j)
    target_up = train_up.iloc[:,-1]
    features_up = train_up.iloc[:,:-1]

    li_model = [LogisticRegression_model(features_up, target_up),
                CatBoostClassifier_model(features_up, target_up),
                RandomForestClassifier_model(features_up, target_up),
                LGBMClassifier_model(features_up, target_up)]
    for mod in li_model:
      model, result = mod
      if result > best_F1:
        best_F1 = result
        best_xN = j
        best_model = model

  return best_xN, best_F1, best_model, best_target_up, best_features_up
```


```python
%%time
best_xN, best_F1, best_model, best_target_up, best_features_up = search_best_xN_F1(train, 1, 5)
```

    Learning rate set to 0.5
    0:	learn: 0.3874567	total: 5.38ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4887235	total: 3.13ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3769760	total: 3.06ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3769760	total: 3.06ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4887235	total: 3.09ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3874567	total: 3.3ms	remaining: 3.3ms
    1:	learn: 0.2072654	total: 6.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4887235	total: 9.71ms	remaining: 9.71ms
    1:	learn: 0.2989211	total: 14.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3769760	total: 3.19ms	remaining: 3.19ms
    1:	learn: 0.1742880	total: 7.34ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3769760	total: 3.04ms	remaining: 3.04ms
    1:	learn: 0.1814913	total: 6.67ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4887235	total: 3.01ms	remaining: 3.01ms
    1:	learn: 0.2964019	total: 6.28ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3917440	total: 3ms	remaining: 6ms
    1:	learn: 0.2102537	total: 6.15ms	remaining: 3.07ms
    2:	learn: 0.1620062	total: 9.25ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.4918108	total: 3.02ms	remaining: 6.04ms
    1:	learn: 0.3014960	total: 6.25ms	remaining: 3.13ms
    2:	learn: 0.2132144	total: 9.41ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3814240	total: 2.99ms	remaining: 5.98ms
    1:	learn: 0.1775573	total: 6.22ms	remaining: 3.11ms
    2:	learn: 0.0805021	total: 9.28ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3814240	total: 3ms	remaining: 6ms
    1:	learn: 0.1845270	total: 7.35ms	remaining: 3.67ms
    2:	learn: 0.0694514	total: 12.7ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.4918108	total: 2.97ms	remaining: 5.93ms
    1:	learn: 0.2994562	total: 6.13ms	remaining: 3.07ms
    2:	learn: 0.2594236	total: 9.64ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4379146	total: 3ms	remaining: 9ms
    1:	learn: 0.3219281	total: 6.15ms	remaining: 6.15ms
    2:	learn: 0.2278421	total: 9.27ms	remaining: 3.09ms
    3:	learn: 0.1553377	total: 12.4ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.5244190	total: 2.99ms	remaining: 8.96ms
    1:	learn: 0.3332554	total: 6.24ms	remaining: 6.24ms
    2:	learn: 0.2883026	total: 9.42ms	remaining: 3.14ms
    3:	learn: 0.1903619	total: 12.5ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4290954	total: 3.04ms	remaining: 9.13ms
    1:	learn: 0.2171022	total: 6.36ms	remaining: 6.36ms
    2:	learn: 0.1119038	total: 9.44ms	remaining: 3.15ms
    3:	learn: 0.0645771	total: 12.6ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4290954	total: 3.01ms	remaining: 9.04ms
    1:	learn: 0.3372279	total: 6.29ms	remaining: 6.29ms
    2:	learn: 0.1367601	total: 9.51ms	remaining: 3.17ms
    3:	learn: 0.0959092	total: 12.7ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.5244189	total: 3.19ms	remaining: 9.56ms
    1:	learn: 0.3360219	total: 6.83ms	remaining: 6.83ms
    2:	learn: 0.2878445	total: 9.89ms	remaining: 3.3ms
    3:	learn: 0.2409004	total: 13ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4727741	total: 3.07ms	remaining: 12.3ms
    1:	learn: 0.3503922	total: 6.46ms	remaining: 9.69ms
    2:	learn: 0.2524081	total: 9.66ms	remaining: 6.44ms
    3:	learn: 0.1798043	total: 12.8ms	remaining: 3.21ms
    4:	learn: 0.1210713	total: 15.9ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.5484077	total: 3.1ms	remaining: 12.4ms
    1:	learn: 0.3629797	total: 6.68ms	remaining: 10ms
    2:	learn: 0.3167787	total: 10.1ms	remaining: 6.74ms
    3:	learn: 0.2444951	total: 13.8ms	remaining: 3.45ms
    4:	learn: 0.2147345	total: 17.1ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4649828	total: 3.03ms	remaining: 12.1ms
    1:	learn: 0.2538715	total: 6.3ms	remaining: 9.45ms
    2:	learn: 0.1411860	total: 9.69ms	remaining: 6.46ms
    3:	learn: 0.0962597	total: 13.2ms	remaining: 3.31ms
    4:	learn: 0.0674149	total: 16.5ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4649829	total: 3.02ms	remaining: 12.1ms
    1:	learn: 0.3665474	total: 6.31ms	remaining: 9.47ms
    2:	learn: 0.2697773	total: 9.49ms	remaining: 6.33ms
    3:	learn: 0.1931976	total: 12.7ms	remaining: 3.16ms
    4:	learn: 0.1066276	total: 15.8ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.5484077	total: 3.04ms	remaining: 12.2ms
    1:	learn: 0.3686790	total: 7.07ms	remaining: 10.6ms
    2:	learn: 0.3144305	total: 10.4ms	remaining: 6.91ms
    3:	learn: 0.2678857	total: 13.5ms	remaining: 3.38ms
    4:	learn: 0.2041571	total: 16.6ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4994374	total: 3.03ms	remaining: 15.1ms
    1:	learn: 0.3766091	total: 6.48ms	remaining: 13ms
    2:	learn: 0.2760436	total: 12.2ms	remaining: 12.2ms
    3:	learn: 0.2020226	total: 15.8ms	remaining: 7.88ms
    4:	learn: 0.1409138	total: 19ms	remaining: 3.79ms
    5:	learn: 0.1053719	total: 22.3ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.5664725	total: 3.46ms	remaining: 17.3ms
    1:	learn: 0.3895115	total: 7.07ms	remaining: 14.1ms
    2:	learn: 0.3416796	total: 10.4ms	remaining: 10.4ms
    3:	learn: 0.2689044	total: 13.6ms	remaining: 6.81ms
    4:	learn: 0.2379468	total: 16.9ms	remaining: 3.37ms
    5:	learn: 0.2026270	total: 20.2ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4924380	total: 3.04ms	remaining: 15.2ms
    1:	learn: 0.2869961	total: 6.38ms	remaining: 12.8ms
    2:	learn: 0.1688174	total: 9.54ms	remaining: 9.54ms
    3:	learn: 0.1175812	total: 12.7ms	remaining: 6.37ms
    4:	learn: 0.0852440	total: 15.8ms	remaining: 3.17ms
    5:	learn: 0.0593424	total: 19ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4924380	total: 3.03ms	remaining: 15.1ms
    1:	learn: 0.3928424	total: 6.17ms	remaining: 12.3ms
    2:	learn: 0.2936033	total: 9.19ms	remaining: 9.19ms
    3:	learn: 0.1819066	total: 13.2ms	remaining: 6.61ms
    4:	learn: 0.1098454	total: 16.3ms	remaining: 3.27ms
    5:	learn: 0.0924312	total: 19.4ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.5664725	total: 3.04ms	remaining: 15.2ms
    1:	learn: 0.3968921	total: 6.37ms	remaining: 12.7ms
    2:	learn: 0.3387946	total: 9.54ms	remaining: 9.54ms
    3:	learn: 0.2916664	total: 12.8ms	remaining: 6.42ms
    4:	learn: 0.2270564	total: 16.1ms	remaining: 3.21ms
    5:	learn: 0.1819411	total: 19.2ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5203317	total: 3.05ms	remaining: 18.3ms
    1:	learn: 0.3999722	total: 6.5ms	remaining: 16.3ms
    2:	learn: 0.2983566	total: 9.87ms	remaining: 13.2ms
    3:	learn: 0.2228173	total: 13.1ms	remaining: 9.86ms
    4:	learn: 0.1593655	total: 16.6ms	remaining: 6.63ms
    5:	learn: 0.1224380	total: 19.7ms	remaining: 3.28ms
    6:	learn: 0.1017280	total: 22.9ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5804958	total: 3.05ms	remaining: 18.3ms
    1:	learn: 0.4127619	total: 6.28ms	remaining: 15.7ms
    2:	learn: 0.3637703	total: 9.41ms	remaining: 12.5ms
    3:	learn: 0.2918483	total: 12.5ms	remaining: 9.38ms
    4:	learn: 0.2631090	total: 15.5ms	remaining: 6.2ms
    5:	learn: 0.2291623	total: 18.8ms	remaining: 3.13ms
    6:	learn: 0.1768774	total: 21.8ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5139744	total: 3.07ms	remaining: 18.4ms
    1:	learn: 0.3163602	total: 6.36ms	remaining: 15.9ms
    2:	learn: 0.2664041	total: 9.51ms	remaining: 12.7ms
    3:	learn: 0.1587557	total: 12.7ms	remaining: 9.53ms
    4:	learn: 0.1197088	total: 15.9ms	remaining: 6.35ms
    5:	learn: 0.0841753	total: 19.1ms	remaining: 3.18ms
    6:	learn: 0.0601247	total: 22.3ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5139744	total: 3.02ms	remaining: 18.1ms
    1:	learn: 0.4158751	total: 6.32ms	remaining: 15.8ms
    2:	learn: 0.3158820	total: 9.54ms	remaining: 12.7ms
    3:	learn: 0.2032968	total: 12.8ms	remaining: 9.59ms
    4:	learn: 0.1281477	total: 15.8ms	remaining: 6.32ms
    5:	learn: 0.0863981	total: 18.8ms	remaining: 3.13ms
    6:	learn: 0.0735206	total: 21.8ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5804958	total: 2.99ms	remaining: 18ms
    1:	learn: 0.4210685	total: 6.23ms	remaining: 15.6ms
    2:	learn: 0.3608612	total: 9.41ms	remaining: 12.5ms
    3:	learn: 0.2940317	total: 12.6ms	remaining: 9.44ms
    4:	learn: 0.2666960	total: 15.7ms	remaining: 6.29ms
    5:	learn: 0.2221624	total: 18.8ms	remaining: 3.14ms
    6:	learn: 0.1943296	total: 21.9ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5370997	total: 3.15ms	remaining: 22.1ms
    1:	learn: 0.4205573	total: 6.61ms	remaining: 19.8ms
    2:	learn: 0.3191177	total: 9.91ms	remaining: 16.5ms
    3:	learn: 0.2424069	total: 13.1ms	remaining: 13.1ms
    4:	learn: 0.1768349	total: 16.5ms	remaining: 9.9ms
    5:	learn: 0.1384213	total: 19.7ms	remaining: 6.57ms
    6:	learn: 0.1163380	total: 22.8ms	remaining: 3.25ms
    7:	learn: 0.1045870	total: 26ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5916659	total: 3.04ms	remaining: 21.3ms
    1:	learn: 0.4330345	total: 6.27ms	remaining: 18.8ms
    2:	learn: 0.3834771	total: 9.39ms	remaining: 15.7ms
    3:	learn: 0.3111127	total: 12.6ms	remaining: 12.6ms
    4:	learn: 0.2403668	total: 15.6ms	remaining: 9.39ms
    5:	learn: 0.2234626	total: 18.8ms	remaining: 6.26ms
    6:	learn: 0.1775128	total: 21.8ms	remaining: 3.11ms
    7:	learn: 0.1664163	total: 24.8ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5312806	total: 3.01ms	remaining: 21.1ms
    1:	learn: 0.3422407	total: 6.28ms	remaining: 18.8ms
    2:	learn: 0.2888765	total: 9.53ms	remaining: 15.9ms
    3:	learn: 0.1797546	total: 12.6ms	remaining: 12.6ms
    4:	learn: 0.1373099	total: 16.1ms	remaining: 9.66ms
    5:	learn: 0.1176440	total: 19.3ms	remaining: 6.42ms
    6:	learn: 0.0836310	total: 22.4ms	remaining: 3.2ms
    7:	learn: 0.0643852	total: 25.6ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5312806	total: 3.02ms	remaining: 21.1ms
    1:	learn: 0.4359435	total: 6.37ms	remaining: 19.1ms
    2:	learn: 0.3364359	total: 9.64ms	remaining: 16.1ms
    3:	learn: 0.2235529	total: 12.9ms	remaining: 12.9ms
    4:	learn: 0.1458589	total: 16.3ms	remaining: 9.79ms
    5:	learn: 0.1249286	total: 19.5ms	remaining: 6.51ms
    6:	learn: 0.0950151	total: 26.2ms	remaining: 3.74ms
    7:	learn: 0.0696696	total: 29.4ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5916659	total: 3.06ms	remaining: 21.4ms
    1:	learn: 0.4418089	total: 6.38ms	remaining: 19.1ms
    2:	learn: 0.3807307	total: 9.56ms	remaining: 15.9ms
    3:	learn: 0.3561246	total: 12.7ms	remaining: 12.7ms
    4:	learn: 0.3243734	total: 15.8ms	remaining: 9.5ms
    5:	learn: 0.2689911	total: 19ms	remaining: 6.32ms
    6:	learn: 0.2121937	total: 22.1ms	remaining: 3.15ms
    7:	learn: 0.1796869	total: 25.3ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5508330	total: 3.03ms	remaining: 24.3ms
    1:	learn: 0.4386522	total: 6.31ms	remaining: 22.1ms
    2:	learn: 0.3382690	total: 9.62ms	remaining: 19.2ms
    3:	learn: 0.2608516	total: 12.8ms	remaining: 16ms
    4:	learn: 0.1934988	total: 15.8ms	remaining: 12.7ms
    5:	learn: 0.1535832	total: 18.7ms	remaining: 9.35ms
    6:	learn: 0.1301495	total: 21.8ms	remaining: 6.24ms
    7:	learn: 0.1172300	total: 25ms	remaining: 3.12ms
    8:	learn: 0.0936393	total: 28.1ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.6007742	total: 3.06ms	remaining: 24.5ms
    1:	learn: 0.4507399	total: 6.31ms	remaining: 22.1ms
    2:	learn: 0.4011111	total: 9.37ms	remaining: 18.7ms
    3:	learn: 0.3287414	total: 12.4ms	remaining: 15.6ms
    4:	learn: 0.3070827	total: 15.4ms	remaining: 12.3ms
    5:	learn: 0.2705312	total: 18.6ms	remaining: 9.32ms
    6:	learn: 0.2327295	total: 21.7ms	remaining: 6.19ms
    7:	learn: 0.2175814	total: 24.7ms	remaining: 3.08ms
    8:	learn: 0.2022621	total: 31ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5454708	total: 3.01ms	remaining: 24.1ms
    1:	learn: 0.3650527	total: 6.22ms	remaining: 21.8ms
    2:	learn: 0.3094633	total: 9.28ms	remaining: 18.6ms
    3:	learn: 0.1996475	total: 12.3ms	remaining: 15.3ms
    4:	learn: 0.1542335	total: 15.4ms	remaining: 12.3ms
    5:	learn: 0.1329043	total: 18.4ms	remaining: 9.21ms
    6:	learn: 0.0962381	total: 21.5ms	remaining: 6.13ms
    7:	learn: 0.0750219	total: 24.4ms	remaining: 3.05ms
    8:	learn: 0.0592030	total: 27.5ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5454708	total: 3.01ms	remaining: 24.1ms
    1:	learn: 0.4534618	total: 6.2ms	remaining: 21.7ms
    2:	learn: 0.3552712	total: 9.31ms	remaining: 18.6ms
    3:	learn: 0.2426869	total: 12.3ms	remaining: 15.4ms
    4:	learn: 0.1629768	total: 15.4ms	remaining: 12.3ms
    5:	learn: 0.1402067	total: 18.4ms	remaining: 9.19ms
    6:	learn: 0.1083967	total: 21.4ms	remaining: 6.11ms
    7:	learn: 0.0806736	total: 24.5ms	remaining: 3.06ms
    8:	learn: 0.0621108	total: 27.6ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.6007742	total: 3.06ms	remaining: 24.4ms
    1:	learn: 0.5156674	total: 6.34ms	remaining: 22.2ms
    2:	learn: 0.4421583	total: 9.76ms	remaining: 19.5ms
    3:	learn: 0.3578801	total: 13.1ms	remaining: 16.4ms
    4:	learn: 0.3271768	total: 19.6ms	remaining: 15.7ms
    5:	learn: 0.2730187	total: 23ms	remaining: 11.5ms
    6:	learn: 0.2347576	total: 26.1ms	remaining: 7.46ms
    7:	learn: 0.2008200	total: 30.2ms	remaining: 3.77ms
    8:	learn: 0.1926872	total: 33.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3687326	total: 6.14ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4085023	total: 5.77ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3134595	total: 5.84ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3737082	total: 6.15ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3738121	total: 5.77ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3687326	total: 5.72ms	remaining: 5.72ms
    1:	learn: 0.1319753	total: 11.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4085023	total: 5.83ms	remaining: 5.83ms
    1:	learn: 0.1385893	total: 12ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3134595	total: 5.84ms	remaining: 5.84ms
    1:	learn: 0.1163909	total: 11.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3737082	total: 5.84ms	remaining: 5.84ms
    1:	learn: 0.1207800	total: 11.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3738121	total: 5.71ms	remaining: 5.71ms
    1:	learn: 0.2081076	total: 11.4ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3733059	total: 5.86ms	remaining: 11.7ms
    1:	learn: 0.2023712	total: 11.7ms	remaining: 5.84ms
    2:	learn: 0.1298783	total: 17.5ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.4124589	total: 5.86ms	remaining: 11.7ms
    1:	learn: 0.1431866	total: 12.1ms	remaining: 6.07ms
    2:	learn: 0.0977029	total: 18.3ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3183403	total: 5.76ms	remaining: 11.5ms
    1:	learn: 0.1204304	total: 11.6ms	remaining: 5.79ms
    2:	learn: 0.0527818	total: 17.2ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3782228	total: 5.8ms	remaining: 11.6ms
    1:	learn: 0.1886806	total: 11.7ms	remaining: 5.87ms
    2:	learn: 0.1210875	total: 17.6ms	remaining: 0us
    Learning rate set to 0.487655
    0:	learn: 0.3783379	total: 6.84ms	remaining: 13.7ms
    1:	learn: 0.2118860	total: 12.4ms	remaining: 6.19ms
    2:	learn: 0.0915054	total: 17.8ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4222830	total: 5.88ms	remaining: 17.6ms
    1:	learn: 0.2692198	total: 11.8ms	remaining: 11.8ms
    2:	learn: 0.1940986	total: 17.6ms	remaining: 5.88ms
    3:	learn: 0.1156107	total: 23.6ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4549689	total: 6.03ms	remaining: 18.1ms
    1:	learn: 0.1962696	total: 11.9ms	remaining: 11.9ms
    2:	learn: 0.1626578	total: 21.7ms	remaining: 7.22ms
    3:	learn: 0.0918781	total: 28.3ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.3714813	total: 9ms	remaining: 27ms
    1:	learn: 0.1708908	total: 15.3ms	remaining: 15.3ms
    2:	learn: 0.1025053	total: 21.6ms	remaining: 7.22ms
    3:	learn: 0.0800686	total: 27.5ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4264839	total: 5.87ms	remaining: 17.6ms
    1:	learn: 0.2305894	total: 11.7ms	remaining: 11.7ms
    2:	learn: 0.1536074	total: 17.8ms	remaining: 5.92ms
    3:	learn: 0.0885662	total: 23.5ms	remaining: 0us
    Learning rate set to 0.37458
    0:	learn: 0.4267278	total: 5.73ms	remaining: 17.2ms
    1:	learn: 0.2559230	total: 11.6ms	remaining: 11.6ms
    2:	learn: 0.1272815	total: 17.3ms	remaining: 5.77ms
    3:	learn: 0.1061512	total: 23ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4591466	total: 5.75ms	remaining: 23ms
    1:	learn: 0.3018212	total: 11.4ms	remaining: 17.2ms
    2:	learn: 0.1952439	total: 17.2ms	remaining: 11.5ms
    3:	learn: 0.1252706	total: 22.9ms	remaining: 5.73ms
    4:	learn: 0.0798127	total: 28.6ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4871259	total: 5.85ms	remaining: 23.4ms
    1:	learn: 0.2419530	total: 12.1ms	remaining: 18.1ms
    2:	learn: 0.2024300	total: 20.1ms	remaining: 13.4ms
    3:	learn: 0.1427848	total: 28.5ms	remaining: 7.13ms
    4:	learn: 0.0954999	total: 34.3ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4126223	total: 5.87ms	remaining: 23.5ms
    1:	learn: 0.2139921	total: 11.8ms	remaining: 17.8ms
    2:	learn: 0.1352682	total: 17.7ms	remaining: 11.8ms
    3:	learn: 0.1068642	total: 23.5ms	remaining: 5.87ms
    4:	learn: 0.0693924	total: 29.2ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4627506	total: 7.24ms	remaining: 29ms
    1:	learn: 0.2688604	total: 14.6ms	remaining: 22ms
    2:	learn: 0.1844536	total: 20.4ms	remaining: 13.6ms
    3:	learn: 0.1119749	total: 28.3ms	remaining: 7.08ms
    4:	learn: 0.0723527	total: 34.2ms	remaining: 0us
    Learning rate set to 0.305265
    0:	learn: 0.4630588	total: 5.89ms	remaining: 23.5ms
    1:	learn: 0.3043761	total: 11.9ms	remaining: 17.8ms
    2:	learn: 0.1659268	total: 17.8ms	remaining: 11.9ms
    3:	learn: 0.1378088	total: 23.7ms	remaining: 5.93ms
    4:	learn: 0.1197751	total: 33.9ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4873356	total: 5.79ms	remaining: 28.9ms
    1:	learn: 0.3315461	total: 11.6ms	remaining: 23.1ms
    2:	learn: 0.2219429	total: 17.4ms	remaining: 17.4ms
    3:	learn: 0.1483712	total: 23.2ms	remaining: 11.6ms
    4:	learn: 0.1086030	total: 29ms	remaining: 5.8ms
    5:	learn: 0.0882450	total: 34.8ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.5117924	total: 5.88ms	remaining: 29.4ms
    1:	learn: 0.3958551	total: 12ms	remaining: 24ms
    2:	learn: 0.2543529	total: 18ms	remaining: 18ms
    3:	learn: 0.1853363	total: 24.4ms	remaining: 12.2ms
    4:	learn: 0.1302610	total: 30.7ms	remaining: 6.14ms
    5:	learn: 0.1063648	total: 37ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4447581	total: 5.79ms	remaining: 29ms
    1:	learn: 0.2980036	total: 11.7ms	remaining: 23.4ms
    2:	learn: 0.1985517	total: 17.5ms	remaining: 17.5ms
    3:	learn: 0.1625099	total: 23.3ms	remaining: 11.7ms
    4:	learn: 0.1059386	total: 29.4ms	remaining: 5.88ms
    5:	learn: 0.0728011	total: 35.3ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4904859	total: 5.76ms	remaining: 28.8ms
    1:	learn: 0.3026994	total: 11.8ms	remaining: 23.6ms
    2:	learn: 0.2133249	total: 17.6ms	remaining: 17.6ms
    3:	learn: 0.1369176	total: 27.9ms	remaining: 14ms
    4:	learn: 0.0911522	total: 34.2ms	remaining: 6.85ms
    5:	learn: 0.0747326	total: 40ms	remaining: 0us
    Learning rate set to 0.258267
    0:	learn: 0.4908287	total: 5.84ms	remaining: 29.2ms
    1:	learn: 0.3373080	total: 12ms	remaining: 23.9ms
    2:	learn: 0.1979245	total: 18.4ms	remaining: 18.4ms
    3:	learn: 0.1644222	total: 24.4ms	remaining: 12.2ms
    4:	learn: 0.1449014	total: 30.3ms	remaining: 6.06ms
    5:	learn: 0.1107593	total: 36.2ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5094535	total: 5.86ms	remaining: 35.1ms
    1:	learn: 0.3579540	total: 12ms	remaining: 29.9ms
    2:	learn: 0.2925658	total: 18ms	remaining: 23.9ms
    3:	learn: 0.1970274	total: 23.8ms	remaining: 17.9ms
    4:	learn: 0.1541019	total: 29.6ms	remaining: 11.8ms
    5:	learn: 0.1268820	total: 37ms	remaining: 6.16ms
    6:	learn: 0.1084011	total: 42.9ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5311845	total: 9.84ms	remaining: 59ms
    1:	learn: 0.4192929	total: 15.7ms	remaining: 39.3ms
    2:	learn: 0.2832658	total: 21.7ms	remaining: 28.9ms
    3:	learn: 0.2109540	total: 27.6ms	remaining: 20.7ms
    4:	learn: 0.1531247	total: 34ms	remaining: 13.6ms
    5:	learn: 0.1266108	total: 40.1ms	remaining: 6.67ms
    6:	learn: 0.1081936	total: 46.2ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.4703431	total: 5.78ms	remaining: 34.7ms
    1:	learn: 0.3269274	total: 14.6ms	remaining: 36.6ms
    2:	learn: 0.2257107	total: 20.3ms	remaining: 27ms
    3:	learn: 0.1858912	total: 26.1ms	remaining: 19.6ms
    4:	learn: 0.1534738	total: 31.9ms	remaining: 12.7ms
    5:	learn: 0.1053056	total: 37.6ms	remaining: 6.27ms
    6:	learn: 0.0730870	total: 43.6ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5122421	total: 5.9ms	remaining: 35.4ms
    1:	learn: 0.3322628	total: 12ms	remaining: 29.9ms
    2:	learn: 0.2418215	total: 17.9ms	remaining: 23.8ms
    3:	learn: 0.1964761	total: 23.7ms	remaining: 17.8ms
    4:	learn: 0.1560740	total: 30.9ms	remaining: 12.4ms
    5:	learn: 0.1338105	total: 39.1ms	remaining: 6.52ms
    6:	learn: 0.0983041	total: 45.2ms	remaining: 0us
    Learning rate set to 0.224222
    0:	learn: 0.5125952	total: 5.82ms	remaining: 34.9ms
    1:	learn: 0.3655446	total: 11.8ms	remaining: 29.4ms
    2:	learn: 0.2270855	total: 17.6ms	remaining: 23.5ms
    3:	learn: 0.1682268	total: 23.3ms	remaining: 17.5ms
    4:	learn: 0.1368417	total: 29.1ms	remaining: 11.6ms
    5:	learn: 0.1046010	total: 35.1ms	remaining: 5.85ms
    6:	learn: 0.0964274	total: 42ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5272136	total: 6.14ms	remaining: 43ms
    1:	learn: 0.3812190	total: 12.2ms	remaining: 36.7ms
    2:	learn: 0.3145462	total: 18.3ms	remaining: 30.4ms
    3:	learn: 0.2522111	total: 24.8ms	remaining: 24.8ms
    4:	learn: 0.1990209	total: 30.9ms	remaining: 18.5ms
    5:	learn: 0.1478932	total: 36.7ms	remaining: 12.2ms
    6:	learn: 0.1245756	total: 42.6ms	remaining: 6.08ms
    7:	learn: 0.1082687	total: 48.6ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5467775	total: 5.97ms	remaining: 41.8ms
    1:	learn: 0.4395080	total: 12.4ms	remaining: 37.1ms
    2:	learn: 0.3087370	total: 19.1ms	remaining: 31.8ms
    3:	learn: 0.2343590	total: 25.5ms	remaining: 25.5ms
    4:	learn: 0.1746057	total: 31.6ms	remaining: 19ms
    5:	learn: 0.1562370	total: 37.7ms	remaining: 12.6ms
    6:	learn: 0.1357063	total: 43.7ms	remaining: 6.24ms
    7:	learn: 0.1307982	total: 49.6ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.4911183	total: 11ms	remaining: 77ms
    1:	learn: 0.3522933	total: 16.9ms	remaining: 50.7ms
    2:	learn: 0.2506195	total: 22.8ms	remaining: 38ms
    3:	learn: 0.2077772	total: 28.6ms	remaining: 28.6ms
    4:	learn: 0.1723342	total: 34.5ms	remaining: 20.7ms
    5:	learn: 0.1212791	total: 40.3ms	remaining: 13.4ms
    6:	learn: 0.0859352	total: 46ms	remaining: 6.58ms
    7:	learn: 0.0792402	total: 51.8ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5297179	total: 7.64ms	remaining: 53.5ms
    1:	learn: 0.3580311	total: 17.4ms	remaining: 52.2ms
    2:	learn: 0.2665245	total: 25.1ms	remaining: 41.9ms
    3:	learn: 0.2186155	total: 31.2ms	remaining: 31.2ms
    4:	learn: 0.1756736	total: 39.1ms	remaining: 23.5ms
    5:	learn: 0.1509600	total: 49.3ms	remaining: 16.4ms
    6:	learn: 0.1147877	total: 55.4ms	remaining: 7.92ms
    7:	learn: 0.0841388	total: 60.9ms	remaining: 0us
    Learning rate set to 0.198381
    0:	learn: 0.5300656	total: 6.09ms	remaining: 42.6ms
    1:	learn: 0.3898475	total: 16.1ms	remaining: 48.2ms
    2:	learn: 0.2535216	total: 23.1ms	remaining: 38.5ms
    3:	learn: 0.1916906	total: 29ms	remaining: 29ms
    4:	learn: 0.1713212	total: 34.9ms	remaining: 21ms
    5:	learn: 0.1317770	total: 40.8ms	remaining: 13.6ms
    6:	learn: 0.1192290	total: 46.6ms	remaining: 6.66ms
    7:	learn: 0.1054815	total: 52.3ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5417741	total: 6.29ms	remaining: 50.3ms
    1:	learn: 0.4017015	total: 12.4ms	remaining: 43.6ms
    2:	learn: 0.3346179	total: 22.4ms	remaining: 44.7ms
    3:	learn: 0.2710371	total: 29.5ms	remaining: 36.8ms
    4:	learn: 0.2158501	total: 35.5ms	remaining: 28.4ms
    5:	learn: 0.1633339	total: 44.1ms	remaining: 22.1ms
    6:	learn: 0.1386016	total: 58.9ms	remaining: 16.8ms
    7:	learn: 0.1209475	total: 65.6ms	remaining: 8.19ms
    8:	learn: 0.1055584	total: 71.5ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5595752	total: 8.24ms	remaining: 65.9ms
    1:	learn: 0.4570433	total: 14ms	remaining: 49ms
    2:	learn: 0.3976434	total: 19.9ms	remaining: 39.7ms
    3:	learn: 0.3067959	total: 25.7ms	remaining: 32.1ms
    4:	learn: 0.2325909	total: 31.6ms	remaining: 25.3ms
    5:	learn: 0.2064148	total: 37.4ms	remaining: 18.7ms
    6:	learn: 0.1765689	total: 43.2ms	remaining: 12.4ms
    7:	learn: 0.1695952	total: 48.9ms	remaining: 6.11ms
    8:	learn: 0.1506611	total: 54.6ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5082978	total: 11ms	remaining: 87.8ms
    1:	learn: 0.3745776	total: 17.5ms	remaining: 61.3ms
    2:	learn: 0.2733960	total: 23.3ms	remaining: 46.7ms
    3:	learn: 0.2282307	total: 29.2ms	remaining: 36.5ms
    4:	learn: 0.1902691	total: 35.6ms	remaining: 28.5ms
    5:	learn: 0.1368342	total: 41.5ms	remaining: 20.8ms
    6:	learn: 0.0987471	total: 47.7ms	remaining: 13.6ms
    7:	learn: 0.0911943	total: 53.6ms	remaining: 6.7ms
    8:	learn: 0.0733529	total: 59.5ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5440480	total: 6.1ms	remaining: 48.8ms
    1:	learn: 0.3805504	total: 16.5ms	remaining: 57.9ms
    2:	learn: 0.2890573	total: 24.4ms	remaining: 48.8ms
    3:	learn: 0.2392342	total: 32.5ms	remaining: 40.6ms
    4:	learn: 0.1942492	total: 38.3ms	remaining: 30.7ms
    5:	learn: 0.1673697	total: 44.1ms	remaining: 22ms
    6:	learn: 0.1263784	total: 49.9ms	remaining: 14.3ms
    7:	learn: 0.0945830	total: 55.7ms	remaining: 6.96ms
    8:	learn: 0.0828259	total: 61.5ms	remaining: 0us
    Learning rate set to 0.178071
    0:	learn: 0.5443887	total: 9.36ms	remaining: 74.9ms
    1:	learn: 0.4108915	total: 14.8ms	remaining: 51.9ms
    2:	learn: 0.2774495	total: 20.1ms	remaining: 40.3ms
    3:	learn: 0.2489623	total: 25.4ms	remaining: 31.8ms
    4:	learn: 0.1992666	total: 32.3ms	remaining: 25.9ms
    5:	learn: 0.1624689	total: 38.4ms	remaining: 19.2ms
    6:	learn: 0.1455056	total: 44.7ms	remaining: 12.8ms
    7:	learn: 0.1321838	total: 51.1ms	remaining: 6.39ms
    8:	learn: 0.1132165	total: 56.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4340786	total: 6.4ms	remaining: 6.4ms
    1:	learn: 0.3225041	total: 13.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3598322	total: 6.27ms	remaining: 6.27ms
    1:	learn: 0.2570089	total: 12.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3378303	total: 6.31ms	remaining: 6.31ms
    1:	learn: 0.2426974	total: 12.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4681924	total: 6.28ms	remaining: 6.28ms
    1:	learn: 0.3211962	total: 12.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4210090	total: 6.39ms	remaining: 6.39ms
    1:	learn: 0.3360956	total: 14.4ms	remaining: 0us
    Learning rate set to 0.361604
    0:	learn: 0.4828970	total: 6.58ms	remaining: 39.5ms
    1:	learn: 0.4088387	total: 13.2ms	remaining: 32.9ms
    2:	learn: 0.3138437	total: 19.7ms	remaining: 26.3ms
    3:	learn: 0.2943544	total: 26.3ms	remaining: 19.7ms
    4:	learn: 0.2657869	total: 32.8ms	remaining: 13.1ms
    5:	learn: 0.2331067	total: 39.4ms	remaining: 6.56ms
    6:	learn: 0.2076875	total: 45.9ms	remaining: 0us
    Learning rate set to 0.361604
    0:	learn: 0.4204489	total: 6.54ms	remaining: 39.2ms
    1:	learn: 0.2898162	total: 13.1ms	remaining: 32.8ms
    2:	learn: 0.1953811	total: 19.9ms	remaining: 26.5ms
    3:	learn: 0.1733414	total: 26.4ms	remaining: 19.8ms
    4:	learn: 0.1552644	total: 33ms	remaining: 13.2ms
    5:	learn: 0.1274584	total: 39.8ms	remaining: 6.64ms
    6:	learn: 0.1180919	total: 46.4ms	remaining: 0us
    Learning rate set to 0.361604
    0:	learn: 0.4014685	total: 6.35ms	remaining: 38.1ms
    1:	learn: 0.2794903	total: 12.9ms	remaining: 32.2ms
    2:	learn: 0.2245927	total: 19.6ms	remaining: 26.1ms
    3:	learn: 0.1845451	total: 26.6ms	remaining: 19.9ms
    4:	learn: 0.1590101	total: 33.2ms	remaining: 13.3ms
    5:	learn: 0.1546747	total: 40.1ms	remaining: 6.68ms
    6:	learn: 0.1473609	total: 46.7ms	remaining: 0us
    Learning rate set to 0.361604
    0:	learn: 0.5114444	total: 6.23ms	remaining: 37.4ms
    1:	learn: 0.3611616	total: 12.8ms	remaining: 31.9ms
    2:	learn: 0.3307290	total: 19.2ms	remaining: 25.7ms
    3:	learn: 0.2835113	total: 25.7ms	remaining: 19.3ms
    4:	learn: 0.2387358	total: 32.1ms	remaining: 12.9ms
    5:	learn: 0.2149259	total: 38.5ms	remaining: 6.42ms
    6:	learn: 0.1940067	total: 45.3ms	remaining: 0us
    Learning rate set to 0.361604
    0:	learn: 0.4722405	total: 6.61ms	remaining: 39.6ms
    1:	learn: 0.3804731	total: 13.3ms	remaining: 33.3ms
    2:	learn: 0.3338766	total: 19.8ms	remaining: 26.4ms
    3:	learn: 0.2608374	total: 26.2ms	remaining: 19.6ms
    4:	learn: 0.2253355	total: 32.6ms	remaining: 13ms
    5:	learn: 0.2074415	total: 39.1ms	remaining: 6.51ms
    6:	learn: 0.1899299	total: 46.1ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.5115418	total: 13ms	remaining: 104ms
    1:	learn: 0.3988167	total: 26.2ms	remaining: 91.7ms
    2:	learn: 0.3167497	total: 39.3ms	remaining: 78.6ms
    3:	learn: 0.2564162	total: 51.9ms	remaining: 64.8ms
    4:	learn: 0.2163451	total: 64.9ms	remaining: 51.9ms
    5:	learn: 0.1736979	total: 77.9ms	remaining: 39ms
    6:	learn: 0.1598801	total: 90.9ms	remaining: 26ms
    7:	learn: 0.1491535	total: 104ms	remaining: 13ms
    8:	learn: 0.1299029	total: 117ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.4432307	total: 13ms	remaining: 104ms
    1:	learn: 0.3196249	total: 26.1ms	remaining: 91.5ms
    2:	learn: 0.2493942	total: 38.9ms	remaining: 77.7ms
    3:	learn: 0.1809077	total: 52ms	remaining: 65.1ms
    4:	learn: 0.1254696	total: 64.8ms	remaining: 51.8ms
    5:	learn: 0.1134700	total: 77.8ms	remaining: 38.9ms
    6:	learn: 0.1044341	total: 90.8ms	remaining: 25.9ms
    7:	learn: 0.0926572	total: 104ms	remaining: 13ms
    8:	learn: 0.0798139	total: 117ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.4256072	total: 12.8ms	remaining: 102ms
    1:	learn: 0.3249642	total: 25.9ms	remaining: 90.7ms
    2:	learn: 0.2348372	total: 39.7ms	remaining: 79.3ms
    3:	learn: 0.1890685	total: 52.9ms	remaining: 66.1ms
    4:	learn: 0.1568077	total: 66ms	remaining: 52.8ms
    5:	learn: 0.1297850	total: 79.2ms	remaining: 39.6ms
    6:	learn: 0.1143313	total: 92.3ms	remaining: 26.4ms
    7:	learn: 0.1076911	total: 109ms	remaining: 13.6ms
    8:	learn: 0.0974437	total: 130ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.5174543	total: 13ms	remaining: 104ms
    1:	learn: 0.4219973	total: 29ms	remaining: 101ms
    2:	learn: 0.3228421	total: 41.7ms	remaining: 83.3ms
    3:	learn: 0.2502789	total: 54.1ms	remaining: 67.7ms
    4:	learn: 0.2287824	total: 66.9ms	remaining: 53.5ms
    5:	learn: 0.1844018	total: 79.8ms	remaining: 39.9ms
    6:	learn: 0.1591828	total: 92.8ms	remaining: 26.5ms
    7:	learn: 0.1256286	total: 106ms	remaining: 13.2ms
    8:	learn: 0.1051937	total: 119ms	remaining: 0us
    Learning rate set to 0.287175
    0:	learn: 0.4929337	total: 12.9ms	remaining: 103ms
    1:	learn: 0.3824131	total: 26.1ms	remaining: 91.4ms
    2:	learn: 0.3032431	total: 38.9ms	remaining: 77.9ms
    3:	learn: 0.2632034	total: 52ms	remaining: 65ms
    4:	learn: 0.2372607	total: 65.1ms	remaining: 52.1ms
    5:	learn: 0.1913073	total: 81ms	remaining: 40.5ms
    6:	learn: 0.1747878	total: 94.5ms	remaining: 27ms
    7:	learn: 0.1490471	total: 108ms	remaining: 13.4ms
    8:	learn: 0.1431979	total: 121ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4273694	total: 12.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3402595	total: 12.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3179122	total: 13.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4406094	total: 13.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4044032	total: 12.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4273694	total: 12.8ms	remaining: 25.6ms
    1:	learn: 0.2623381	total: 31.8ms	remaining: 15.9ms
    2:	learn: 0.1982561	total: 44.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3402595	total: 17.4ms	remaining: 34.8ms
    1:	learn: 0.2207023	total: 34ms	remaining: 17ms
    2:	learn: 0.1532746	total: 46.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3179122	total: 16.3ms	remaining: 32.5ms
    1:	learn: 0.2406073	total: 29.8ms	remaining: 14.9ms
    2:	learn: 0.1734675	total: 42.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4406094	total: 13.1ms	remaining: 26.2ms
    1:	learn: 0.3075292	total: 35.3ms	remaining: 17.6ms
    2:	learn: 0.2309020	total: 49.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4044032	total: 13.3ms	remaining: 26.7ms
    1:	learn: 0.2396134	total: 27.8ms	remaining: 13.9ms
    2:	learn: 0.1857337	total: 44.9ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4297452	total: 13.4ms	remaining: 53.6ms
    1:	learn: 0.2643518	total: 31.7ms	remaining: 47.6ms
    2:	learn: 0.2004040	total: 45.7ms	remaining: 30.4ms
    3:	learn: 0.1760991	total: 59ms	remaining: 14.7ms
    4:	learn: 0.1409892	total: 72.7ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.3430465	total: 13.7ms	remaining: 54.7ms
    1:	learn: 0.2223479	total: 27ms	remaining: 40.5ms
    2:	learn: 0.1547387	total: 40ms	remaining: 26.7ms
    3:	learn: 0.1161741	total: 53ms	remaining: 13.2ms
    4:	learn: 0.0777446	total: 65.9ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.3207925	total: 13.1ms	remaining: 52.4ms
    1:	learn: 0.2419831	total: 26.7ms	remaining: 40.1ms
    2:	learn: 0.1745516	total: 40ms	remaining: 26.7ms
    3:	learn: 0.0987019	total: 53.6ms	remaining: 13.4ms
    4:	learn: 0.0710848	total: 67.3ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4427272	total: 13.2ms	remaining: 52.7ms
    1:	learn: 0.3091558	total: 26.7ms	remaining: 40.1ms
    2:	learn: 0.2323298	total: 43ms	remaining: 28.7ms
    3:	learn: 0.2076063	total: 56.1ms	remaining: 14ms
    4:	learn: 0.1650643	total: 69.1ms	remaining: 0us
    Learning rate set to 0.492303
    0:	learn: 0.4068521	total: 13.1ms	remaining: 52.5ms
    1:	learn: 0.2412696	total: 27.2ms	remaining: 40.7ms
    2:	learn: 0.1873985	total: 50.4ms	remaining: 33.6ms
    3:	learn: 0.1485440	total: 67ms	remaining: 16.7ms
    4:	learn: 0.1236189	total: 80.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4255151	total: 34.3ms	remaining: 68.5ms
    1:	learn: 0.3339575	total: 63.2ms	remaining: 31.6ms
    2:	learn: 0.2579941	total: 94.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3980661	total: 29.9ms	remaining: 59.9ms
    1:	learn: 0.3302194	total: 60.5ms	remaining: 30.2ms
    2:	learn: 0.2977149	total: 91.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4118307	total: 29.7ms	remaining: 59.4ms
    1:	learn: 0.2860910	total: 60.1ms	remaining: 30.1ms
    2:	learn: 0.2185654	total: 90.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4104472	total: 29.2ms	remaining: 58.4ms
    1:	learn: 0.3389662	total: 58.7ms	remaining: 29.4ms
    2:	learn: 0.2735439	total: 88.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4210174	total: 29.5ms	remaining: 59ms
    1:	learn: 0.3203818	total: 59.3ms	remaining: 29.6ms
    2:	learn: 0.2867807	total: 88.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4255151	total: 30.7ms	remaining: 123ms
    1:	learn: 0.3339575	total: 73.6ms	remaining: 110ms
    2:	learn: 0.2579941	total: 103ms	remaining: 68.8ms
    3:	learn: 0.2265374	total: 133ms	remaining: 33.2ms
    4:	learn: 0.1907496	total: 168ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3980661	total: 29.4ms	remaining: 118ms
    1:	learn: 0.3302194	total: 59ms	remaining: 88.6ms
    2:	learn: 0.2977149	total: 91.2ms	remaining: 60.8ms
    3:	learn: 0.2701533	total: 121ms	remaining: 30.2ms
    4:	learn: 0.2358774	total: 151ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4118307	total: 30ms	remaining: 120ms
    1:	learn: 0.2860910	total: 63.1ms	remaining: 94.7ms
    2:	learn: 0.2185654	total: 92.9ms	remaining: 62ms
    3:	learn: 0.1831630	total: 123ms	remaining: 30.8ms
    4:	learn: 0.1580054	total: 153ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4104472	total: 29.3ms	remaining: 117ms
    1:	learn: 0.3389662	total: 58.9ms	remaining: 88.4ms
    2:	learn: 0.2735439	total: 88.3ms	remaining: 58.9ms
    3:	learn: 0.2382852	total: 118ms	remaining: 29.5ms
    4:	learn: 0.1996583	total: 147ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4210174	total: 29.5ms	remaining: 118ms
    1:	learn: 0.3203818	total: 59ms	remaining: 88.5ms
    2:	learn: 0.2867807	total: 88.4ms	remaining: 58.9ms
    3:	learn: 0.2671989	total: 117ms	remaining: 29.4ms
    4:	learn: 0.2436370	total: 147ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4160051	total: 29.6ms	remaining: 118ms
    1:	learn: 0.3213547	total: 59.4ms	remaining: 89ms
    2:	learn: 0.2786127	total: 89ms	remaining: 59.4ms
    3:	learn: 0.2334427	total: 119ms	remaining: 29.8ms
    4:	learn: 0.2056394	total: 150ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5374021	total: 3.28ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6132650	total: 3.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5927302	total: 3.19ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3268632	total: 3.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4651965	total: 3.18ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5374021	total: 3.18ms	remaining: 3.18ms
    1:	learn: 0.3308241	total: 6.53ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6132650	total: 3.15ms	remaining: 3.15ms
    1:	learn: 0.3566013	total: 6.54ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5927302	total: 3.25ms	remaining: 3.25ms
    1:	learn: 0.5167434	total: 6.78ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3268632	total: 3.19ms	remaining: 3.19ms
    1:	learn: 0.1783167	total: 6.29ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4651965	total: 3.17ms	remaining: 3.17ms
    1:	learn: 0.3350434	total: 6.61ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5374021	total: 3.29ms	remaining: 6.57ms
    1:	learn: 0.3308241	total: 7.75ms	remaining: 3.88ms
    2:	learn: 0.2534944	total: 14ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6132650	total: 6.06ms	remaining: 12.1ms
    1:	learn: 0.3566013	total: 9.72ms	remaining: 4.86ms
    2:	learn: 0.2401862	total: 13.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5927302	total: 3.92ms	remaining: 7.84ms
    1:	learn: 0.5167434	total: 7.44ms	remaining: 3.72ms
    2:	learn: 0.4211195	total: 10.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3268632	total: 3.57ms	remaining: 7.14ms
    1:	learn: 0.1783167	total: 7.07ms	remaining: 3.54ms
    2:	learn: 0.1274072	total: 10.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4651965	total: 3.25ms	remaining: 6.49ms
    1:	learn: 0.3350434	total: 6.81ms	remaining: 3.4ms
    2:	learn: 0.2689102	total: 10.4ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.5603611	total: 3.24ms	remaining: 9.73ms
    1:	learn: 0.3815383	total: 6.51ms	remaining: 6.51ms
    2:	learn: 0.3175322	total: 10.5ms	remaining: 3.51ms
    3:	learn: 0.2597000	total: 14ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.6253773	total: 3.17ms	remaining: 9.52ms
    1:	learn: 0.3944547	total: 6.56ms	remaining: 6.56ms
    2:	learn: 0.2789559	total: 9.86ms	remaining: 3.29ms
    3:	learn: 0.2211049	total: 13.3ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.6078725	total: 3.27ms	remaining: 9.82ms
    1:	learn: 0.5320631	total: 6.82ms	remaining: 6.82ms
    2:	learn: 0.4213316	total: 10.2ms	remaining: 3.41ms
    3:	learn: 0.3203617	total: 13.6ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.3716648	total: 3.27ms	remaining: 9.8ms
    1:	learn: 0.2192639	total: 7.05ms	remaining: 7.05ms
    2:	learn: 0.1679582	total: 10.5ms	remaining: 3.51ms
    3:	learn: 0.1368376	total: 13.9ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.4977914	total: 3.22ms	remaining: 9.66ms
    1:	learn: 0.3610601	total: 6.8ms	remaining: 6.8ms
    2:	learn: 0.2913735	total: 10.2ms	remaining: 3.38ms
    3:	learn: 0.1903392	total: 13.4ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.5791557	total: 3.95ms	remaining: 15.8ms
    1:	learn: 0.4729048	total: 7.38ms	remaining: 11.1ms
    2:	learn: 0.3893356	total: 11.8ms	remaining: 7.83ms
    3:	learn: 0.3271746	total: 15ms	remaining: 3.74ms
    4:	learn: 0.2890328	total: 18.3ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.6351790	total: 3.12ms	remaining: 12.5ms
    1:	learn: 0.4286474	total: 6.48ms	remaining: 9.72ms
    2:	learn: 0.3155668	total: 9.9ms	remaining: 6.6ms
    3:	learn: 0.2522660	total: 13.3ms	remaining: 3.33ms
    4:	learn: 0.2411948	total: 16.5ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.6201479	total: 3.19ms	remaining: 12.8ms
    1:	learn: 0.5066665	total: 6.76ms	remaining: 10.1ms
    2:	learn: 0.4420812	total: 10.4ms	remaining: 6.91ms
    3:	learn: 0.3921996	total: 13.8ms	remaining: 3.44ms
    4:	learn: 0.3530057	total: 17.2ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.4112989	total: 3.26ms	remaining: 13.1ms
    1:	learn: 0.2584129	total: 6.78ms	remaining: 10.2ms
    2:	learn: 0.2062174	total: 10.2ms	remaining: 6.81ms
    3:	learn: 0.1641940	total: 13.8ms	remaining: 3.44ms
    4:	learn: 0.1322474	total: 17.2ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.5248495	total: 3.19ms	remaining: 12.8ms
    1:	learn: 0.4105795	total: 6.72ms	remaining: 10.1ms
    2:	learn: 0.3361483	total: 10.3ms	remaining: 6.85ms
    3:	learn: 0.2518945	total: 13.7ms	remaining: 3.42ms
    4:	learn: 0.1945512	total: 17ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.5933518	total: 3.29ms	remaining: 16.5ms
    1:	learn: 0.4918127	total: 6.99ms	remaining: 14ms
    2:	learn: 0.4107371	total: 10.6ms	remaining: 10.6ms
    3:	learn: 0.3491639	total: 14.3ms	remaining: 7.13ms
    4:	learn: 0.3215641	total: 17.9ms	remaining: 3.58ms
    5:	learn: 0.3026117	total: 21.4ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.6425200	total: 3.11ms	remaining: 15.6ms
    1:	learn: 0.4563062	total: 6.51ms	remaining: 13ms
    2:	learn: 0.3467970	total: 9.83ms	remaining: 9.83ms
    3:	learn: 0.2857858	total: 13.2ms	remaining: 6.61ms
    4:	learn: 0.2723525	total: 16.6ms	remaining: 3.32ms
    5:	learn: 0.2494847	total: 20ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.6293679	total: 3.3ms	remaining: 16.5ms
    1:	learn: 0.5235279	total: 7.05ms	remaining: 14.1ms
    2:	learn: 0.4612438	total: 10.5ms	remaining: 10.5ms
    3:	learn: 0.4139749	total: 13.8ms	remaining: 6.91ms
    4:	learn: 0.3833314	total: 17.1ms	remaining: 3.43ms
    5:	learn: 0.3580045	total: 20.4ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.4428058	total: 3.23ms	remaining: 16.2ms
    1:	learn: 0.2923879	total: 6.82ms	remaining: 13.6ms
    2:	learn: 0.2393988	total: 10.3ms	remaining: 10.3ms
    3:	learn: 0.1981761	total: 13.8ms	remaining: 6.89ms
    4:	learn: 0.1623256	total: 17.3ms	remaining: 3.46ms
    5:	learn: 0.1522122	total: 20.7ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.5454405	total: 3.16ms	remaining: 15.8ms
    1:	learn: 0.4336102	total: 6.36ms	remaining: 12.7ms
    2:	learn: 0.3560716	total: 9.65ms	remaining: 9.65ms
    3:	learn: 0.2719922	total: 12.9ms	remaining: 6.47ms
    4:	learn: 0.2167540	total: 16.2ms	remaining: 3.23ms
    5:	learn: 0.1992255	total: 19.5ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.6043794	total: 3.15ms	remaining: 18.9ms
    1:	learn: 0.5081277	total: 6.59ms	remaining: 16.5ms
    2:	learn: 0.4513466	total: 9.89ms	remaining: 13.2ms
    3:	learn: 0.3837649	total: 13.3ms	remaining: 9.96ms
    4:	learn: 0.3502309	total: 16.6ms	remaining: 6.63ms
    5:	learn: 0.3328778	total: 19.9ms	remaining: 3.31ms
    6:	learn: 0.2715238	total: 23.1ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.6482006	total: 3.23ms	remaining: 19.4ms
    1:	learn: 0.4788563	total: 6.67ms	remaining: 16.7ms
    2:	learn: 0.4332583	total: 9.92ms	remaining: 13.2ms
    3:	learn: 0.3697205	total: 13.2ms	remaining: 9.93ms
    4:	learn: 0.3543942	total: 16.6ms	remaining: 6.64ms
    5:	learn: 0.3248400	total: 20ms	remaining: 3.33ms
    6:	learn: 0.3052501	total: 23.4ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.6364971	total: 3.18ms	remaining: 19.1ms
    1:	learn: 0.5378636	total: 6.67ms	remaining: 16.7ms
    2:	learn: 0.4778559	total: 10ms	remaining: 13.4ms
    3:	learn: 0.4085716	total: 13.4ms	remaining: 10.1ms
    4:	learn: 0.3808808	total: 16.8ms	remaining: 6.72ms
    5:	learn: 0.3584436	total: 22.9ms	remaining: 3.82ms
    6:	learn: 0.3212070	total: 28ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.4681274	total: 3.24ms	remaining: 19.5ms
    1:	learn: 0.3218900	total: 6.73ms	remaining: 16.8ms
    2:	learn: 0.2684559	total: 10.2ms	remaining: 13.6ms
    3:	learn: 0.2277682	total: 13.5ms	remaining: 10.1ms
    4:	learn: 0.1892661	total: 16.9ms	remaining: 6.74ms
    5:	learn: 0.1783809	total: 20.2ms	remaining: 3.37ms
    6:	learn: 0.1494138	total: 23.6ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.5615259	total: 3.16ms	remaining: 19ms
    1:	learn: 0.4537635	total: 6.58ms	remaining: 16.4ms
    2:	learn: 0.3748850	total: 9.88ms	remaining: 13.2ms
    3:	learn: 0.2907509	total: 13.1ms	remaining: 9.79ms
    4:	learn: 0.2703588	total: 16.5ms	remaining: 6.61ms
    5:	learn: 0.2412867	total: 19.7ms	remaining: 3.28ms
    6:	learn: 0.2109530	total: 23.1ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.6131735	total: 3.21ms	remaining: 22.5ms
    1:	learn: 0.5221697	total: 6.61ms	remaining: 19.8ms
    2:	learn: 0.4665180	total: 10ms	remaining: 16.7ms
    3:	learn: 0.4008429	total: 13.2ms	remaining: 13.2ms
    4:	learn: 0.3655621	total: 16.4ms	remaining: 9.87ms
    5:	learn: 0.3481652	total: 19.8ms	remaining: 6.59ms
    6:	learn: 0.2894878	total: 23ms	remaining: 3.28ms
    7:	learn: 0.2694089	total: 26.1ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.6527098	total: 3.3ms	remaining: 23.1ms
    1:	learn: 0.4974958	total: 6.67ms	remaining: 20ms
    2:	learn: 0.4526635	total: 9.97ms	remaining: 16.6ms
    3:	learn: 0.3914990	total: 13.1ms	remaining: 13.1ms
    4:	learn: 0.3745121	total: 16.4ms	remaining: 9.86ms
    5:	learn: 0.3459420	total: 19.6ms	remaining: 6.54ms
    6:	learn: 0.3276347	total: 24.4ms	remaining: 3.49ms
    7:	learn: 0.2688059	total: 33.1ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.6421661	total: 3.25ms	remaining: 22.7ms
    1:	learn: 0.5500798	total: 6.7ms	remaining: 20.1ms
    2:	learn: 0.4923381	total: 10.1ms	remaining: 16.8ms
    3:	learn: 0.4716283	total: 13.5ms	remaining: 13.5ms
    4:	learn: 0.4340763	total: 16.8ms	remaining: 10.1ms
    5:	learn: 0.4074860	total: 20.2ms	remaining: 6.72ms
    6:	learn: 0.3655008	total: 23.5ms	remaining: 3.35ms
    7:	learn: 0.3350947	total: 27ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.4888101	total: 3.18ms	remaining: 22.2ms
    1:	learn: 0.3476045	total: 6.5ms	remaining: 19.5ms
    2:	learn: 0.2940853	total: 9.68ms	remaining: 16.1ms
    3:	learn: 0.2538056	total: 12.9ms	remaining: 12.9ms
    4:	learn: 0.2135433	total: 16ms	remaining: 9.63ms
    5:	learn: 0.2019467	total: 19.3ms	remaining: 6.43ms
    6:	learn: 0.1709608	total: 22.6ms	remaining: 3.22ms
    7:	learn: 0.1500219	total: 26ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.5744028	total: 3.23ms	remaining: 22.6ms
    1:	learn: 0.4712886	total: 6.69ms	remaining: 20.1ms
    2:	learn: 0.3923094	total: 10ms	remaining: 16.7ms
    3:	learn: 0.3083320	total: 13.2ms	remaining: 13.2ms
    4:	learn: 0.2870585	total: 16.7ms	remaining: 10ms
    5:	learn: 0.2572549	total: 20.1ms	remaining: 6.7ms
    6:	learn: 0.2282264	total: 23.6ms	remaining: 3.37ms
    7:	learn: 0.2211922	total: 28ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.6203447	total: 3.2ms	remaining: 25.6ms
    1:	learn: 0.5645875	total: 6.66ms	remaining: 23.3ms
    2:	learn: 0.5001258	total: 9.96ms	remaining: 19.9ms
    3:	learn: 0.4356436	total: 13.4ms	remaining: 16.7ms
    4:	learn: 0.3952901	total: 20.1ms	remaining: 16.1ms
    5:	learn: 0.3761130	total: 26ms	remaining: 13ms
    6:	learn: 0.3168354	total: 33.8ms	remaining: 9.66ms
    7:	learn: 0.2927091	total: 38.4ms	remaining: 4.8ms
    8:	learn: 0.2529392	total: 41.4ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.6563748	total: 4.2ms	remaining: 33.6ms
    1:	learn: 0.5131272	total: 7.79ms	remaining: 27.3ms
    2:	learn: 0.4693742	total: 11.2ms	remaining: 22.4ms
    3:	learn: 0.4104933	total: 14.4ms	remaining: 18ms
    4:	learn: 0.3923321	total: 17.6ms	remaining: 14.1ms
    5:	learn: 0.3646191	total: 21ms	remaining: 10.5ms
    6:	learn: 0.3472147	total: 24.1ms	remaining: 6.88ms
    7:	learn: 0.3236855	total: 27.2ms	remaining: 3.4ms
    8:	learn: 0.2979591	total: 30.4ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.6467810	total: 3.17ms	remaining: 25.4ms
    1:	learn: 0.6043170	total: 6.19ms	remaining: 21.7ms
    2:	learn: 0.5174476	total: 9.15ms	remaining: 18.3ms
    3:	learn: 0.4952703	total: 12.5ms	remaining: 15.6ms
    4:	learn: 0.4616551	total: 15.9ms	remaining: 12.7ms
    5:	learn: 0.4322036	total: 19.1ms	remaining: 9.56ms
    6:	learn: 0.3905673	total: 22.4ms	remaining: 6.41ms
    7:	learn: 0.3590781	total: 26ms	remaining: 3.25ms
    8:	learn: 0.3444272	total: 29.3ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.5059822	total: 3.18ms	remaining: 25.5ms
    1:	learn: 0.4590037	total: 6.61ms	remaining: 23.1ms
    2:	learn: 0.3900836	total: 10ms	remaining: 20.1ms
    3:	learn: 0.2878539	total: 13.4ms	remaining: 16.8ms
    4:	learn: 0.2379638	total: 16.8ms	remaining: 13.4ms
    5:	learn: 0.2273479	total: 20.2ms	remaining: 10.1ms
    6:	learn: 0.1911808	total: 23.5ms	remaining: 6.73ms
    7:	learn: 0.1692828	total: 30.5ms	remaining: 3.81ms
    8:	learn: 0.1522840	total: 34.4ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.5849316	total: 3.19ms	remaining: 25.6ms
    1:	learn: 0.4865425	total: 6.57ms	remaining: 23ms
    2:	learn: 0.4082911	total: 9.9ms	remaining: 19.8ms
    3:	learn: 0.3247920	total: 13.3ms	remaining: 16.6ms
    4:	learn: 0.3025178	total: 16.5ms	remaining: 13.2ms
    5:	learn: 0.2719225	total: 19.7ms	remaining: 9.86ms
    6:	learn: 0.2436586	total: 23ms	remaining: 6.56ms
    7:	learn: 0.2361041	total: 25.9ms	remaining: 3.24ms
    8:	learn: 0.2189060	total: 29.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3544893	total: 6.36ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4715368	total: 6.13ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4529119	total: 6.28ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3397750	total: 6.25ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4200767	total: 6.51ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3544893	total: 8.55ms	remaining: 8.55ms
    1:	learn: 0.2854533	total: 14.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4715368	total: 6.2ms	remaining: 6.2ms
    1:	learn: 0.2826452	total: 12.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4529119	total: 6.36ms	remaining: 6.36ms
    1:	learn: 0.2930221	total: 12.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3397750	total: 6.62ms	remaining: 6.62ms
    1:	learn: 0.2273127	total: 13.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4200767	total: 6.24ms	remaining: 6.24ms
    1:	learn: 0.2122271	total: 12.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3544893	total: 6.27ms	remaining: 12.5ms
    1:	learn: 0.2854533	total: 13.2ms	remaining: 6.6ms
    2:	learn: 0.2096805	total: 19.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4715368	total: 6.09ms	remaining: 12.2ms
    1:	learn: 0.2826452	total: 12.2ms	remaining: 6.12ms
    2:	learn: 0.1626183	total: 18.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4529119	total: 6.35ms	remaining: 12.7ms
    1:	learn: 0.2930221	total: 14.4ms	remaining: 7.18ms
    2:	learn: 0.1969633	total: 22.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3397750	total: 6.29ms	remaining: 12.6ms
    1:	learn: 0.2273127	total: 12.7ms	remaining: 6.34ms
    2:	learn: 0.1005881	total: 22.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4200767	total: 6.28ms	remaining: 12.6ms
    1:	learn: 0.2122271	total: 12.7ms	remaining: 6.36ms
    2:	learn: 0.1304550	total: 19.1ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.3984657	total: 6.3ms	remaining: 18.9ms
    1:	learn: 0.2541139	total: 12.6ms	remaining: 12.6ms
    2:	learn: 0.2051312	total: 19ms	remaining: 6.33ms
    3:	learn: 0.1404050	total: 25.3ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.5031011	total: 6.13ms	remaining: 18.4ms
    1:	learn: 0.3247113	total: 12.2ms	remaining: 12.2ms
    2:	learn: 0.2138045	total: 18.5ms	remaining: 6.16ms
    3:	learn: 0.1696905	total: 28.2ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.4869112	total: 6.43ms	remaining: 19.3ms
    1:	learn: 0.3379227	total: 13ms	remaining: 13ms
    2:	learn: 0.2372769	total: 21.7ms	remaining: 7.22ms
    3:	learn: 0.1571519	total: 33.5ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.3844786	total: 6.12ms	remaining: 18.4ms
    1:	learn: 0.2707542	total: 12.3ms	remaining: 12.3ms
    2:	learn: 0.1408830	total: 18.7ms	remaining: 6.24ms
    3:	learn: 0.1095990	total: 25ms	remaining: 0us
    Learning rate set to 0.3939
    0:	learn: 0.4587371	total: 6.47ms	remaining: 19.4ms
    1:	learn: 0.2209398	total: 13.1ms	remaining: 13.1ms
    2:	learn: 0.1561497	total: 19.6ms	remaining: 6.53ms
    3:	learn: 0.1119026	total: 25.9ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.4363374	total: 6.41ms	remaining: 25.6ms
    1:	learn: 0.2958643	total: 14.8ms	remaining: 22.2ms
    2:	learn: 0.2485444	total: 21.7ms	remaining: 14.5ms
    3:	learn: 0.2142713	total: 29.4ms	remaining: 7.34ms
    4:	learn: 0.1800076	total: 40ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.5292690	total: 6.15ms	remaining: 24.6ms
    1:	learn: 0.3632600	total: 12.3ms	remaining: 18.5ms
    2:	learn: 0.2800919	total: 18.5ms	remaining: 12.3ms
    3:	learn: 0.2294327	total: 24.8ms	remaining: 6.19ms
    4:	learn: 0.1804816	total: 31ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.5151847	total: 6.25ms	remaining: 25ms
    1:	learn: 0.4141320	total: 12.6ms	remaining: 18.9ms
    2:	learn: 0.3268467	total: 18.8ms	remaining: 12.5ms
    3:	learn: 0.2601556	total: 26.1ms	remaining: 6.53ms
    4:	learn: 0.2098894	total: 32.6ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.4234216	total: 6.36ms	remaining: 25.5ms
    1:	learn: 0.3249137	total: 12.8ms	remaining: 19.1ms
    2:	learn: 0.1839238	total: 19.2ms	remaining: 12.8ms
    3:	learn: 0.1549077	total: 25.5ms	remaining: 6.37ms
    4:	learn: 0.1123324	total: 31.8ms	remaining: 0us
    Learning rate set to 0.321011
    0:	learn: 0.4907958	total: 6.45ms	remaining: 25.8ms
    1:	learn: 0.3448111	total: 13ms	remaining: 19.5ms
    2:	learn: 0.2466377	total: 19.3ms	remaining: 12.9ms
    3:	learn: 0.1968966	total: 30.2ms	remaining: 7.55ms
    4:	learn: 0.1706474	total: 37ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.4659531	total: 6.15ms	remaining: 30.7ms
    1:	learn: 0.3309488	total: 12.4ms	remaining: 24.9ms
    2:	learn: 0.2811544	total: 18.6ms	remaining: 18.6ms
    3:	learn: 0.2451289	total: 24.9ms	remaining: 12.4ms
    4:	learn: 0.2101803	total: 31.1ms	remaining: 6.23ms
    5:	learn: 0.1831364	total: 37.4ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.5492237	total: 6.12ms	remaining: 30.6ms
    1:	learn: 0.3950988	total: 12.8ms	remaining: 25.6ms
    2:	learn: 0.3369015	total: 19ms	remaining: 19ms
    3:	learn: 0.2813040	total: 25ms	remaining: 12.5ms
    4:	learn: 0.2288100	total: 31.1ms	remaining: 6.21ms
    5:	learn: 0.1750666	total: 37.1ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.5367827	total: 6.31ms	remaining: 31.6ms
    1:	learn: 0.4401124	total: 19.2ms	remaining: 38.3ms
    2:	learn: 0.3548717	total: 25.5ms	remaining: 25.5ms
    3:	learn: 0.2872523	total: 32.1ms	remaining: 16ms
    4:	learn: 0.2400534	total: 38.4ms	remaining: 7.69ms
    5:	learn: 0.2155087	total: 45.2ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.4540935	total: 6.35ms	remaining: 31.8ms
    1:	learn: 0.3551777	total: 12.7ms	remaining: 25.5ms
    2:	learn: 0.2132613	total: 22.8ms	remaining: 22.8ms
    3:	learn: 0.1813754	total: 29.3ms	remaining: 14.6ms
    4:	learn: 0.1379314	total: 35.7ms	remaining: 7.14ms
    5:	learn: 0.1062700	total: 42.1ms	remaining: 0us
    Learning rate set to 0.271588
    0:	learn: 0.5152604	total: 6.4ms	remaining: 32ms
    1:	learn: 0.3753221	total: 13.2ms	remaining: 26.4ms
    2:	learn: 0.2753764	total: 19.7ms	remaining: 19.7ms
    3:	learn: 0.1880754	total: 25.8ms	remaining: 12.9ms
    4:	learn: 0.1715195	total: 33.1ms	remaining: 6.63ms
    5:	learn: 0.1507154	total: 42.5ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.4895114	total: 6.5ms	remaining: 39ms
    1:	learn: 0.3605962	total: 13.1ms	remaining: 32.7ms
    2:	learn: 0.3094248	total: 19.5ms	remaining: 25.9ms
    3:	learn: 0.2720633	total: 25.8ms	remaining: 19.4ms
    4:	learn: 0.2411941	total: 32.3ms	remaining: 12.9ms
    5:	learn: 0.2041677	total: 38.9ms	remaining: 6.48ms
    6:	learn: 0.1924458	total: 45.4ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.5648303	total: 6.5ms	remaining: 39ms
    1:	learn: 0.4215367	total: 13.1ms	remaining: 32.8ms
    2:	learn: 0.3643400	total: 19.4ms	remaining: 25.8ms
    3:	learn: 0.3079517	total: 25.6ms	remaining: 19.2ms
    4:	learn: 0.2562872	total: 32.1ms	remaining: 12.8ms
    5:	learn: 0.2014569	total: 42.1ms	remaining: 7.02ms
    6:	learn: 0.1887149	total: 49ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.5536910	total: 13.4ms	remaining: 80.6ms
    1:	learn: 0.4618871	total: 20.5ms	remaining: 51.4ms
    2:	learn: 0.3609259	total: 27ms	remaining: 36ms
    3:	learn: 0.2930000	total: 34.1ms	remaining: 25.6ms
    4:	learn: 0.2495033	total: 41.1ms	remaining: 16.5ms
    5:	learn: 0.2256206	total: 48ms	remaining: 8ms
    6:	learn: 0.2030671	total: 55ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.4786039	total: 7.32ms	remaining: 43.9ms
    1:	learn: 0.3813630	total: 18ms	remaining: 45.1ms
    2:	learn: 0.2403110	total: 24.7ms	remaining: 32.9ms
    3:	learn: 0.2057648	total: 35ms	remaining: 26.2ms
    4:	learn: 0.1621623	total: 44ms	remaining: 17.6ms
    5:	learn: 0.1410542	total: 50.5ms	remaining: 8.42ms
    6:	learn: 0.1246666	total: 56.9ms	remaining: 0us
    Learning rate set to 0.235787
    0:	learn: 0.5344270	total: 6.56ms	remaining: 39.3ms
    1:	learn: 0.4012310	total: 17.7ms	remaining: 44.2ms
    2:	learn: 0.3011659	total: 25.1ms	remaining: 33.4ms
    3:	learn: 0.2669837	total: 31.3ms	remaining: 23.5ms
    4:	learn: 0.2364271	total: 37.5ms	remaining: 15ms
    5:	learn: 0.2089218	total: 43.7ms	remaining: 7.28ms
    6:	learn: 0.1916219	total: 49.9ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.5086260	total: 6.36ms	remaining: 44.5ms
    1:	learn: 0.4479600	total: 12.8ms	remaining: 38.3ms
    2:	learn: 0.3883528	total: 23.9ms	remaining: 39.9ms
    3:	learn: 0.3497169	total: 37.6ms	remaining: 37.6ms
    4:	learn: 0.3193275	total: 43.9ms	remaining: 26.4ms
    5:	learn: 0.2748814	total: 50.2ms	remaining: 16.7ms
    6:	learn: 0.2567015	total: 56.5ms	remaining: 8.07ms
    7:	learn: 0.2334986	total: 63ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.5773375	total: 10.5ms	remaining: 73.7ms
    1:	learn: 0.4437471	total: 17.5ms	remaining: 52.6ms
    2:	learn: 0.3879652	total: 23.9ms	remaining: 39.8ms
    3:	learn: 0.3313441	total: 30.1ms	remaining: 30.1ms
    4:	learn: 0.2803819	total: 36.4ms	remaining: 21.8ms
    5:	learn: 0.2260778	total: 42.6ms	remaining: 14.2ms
    6:	learn: 0.2206761	total: 49.3ms	remaining: 7.04ms
    7:	learn: 0.1960136	total: 55.8ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.5672514	total: 6.33ms	remaining: 44.3ms
    1:	learn: 0.4802960	total: 13.7ms	remaining: 41ms
    2:	learn: 0.3817439	total: 24.2ms	remaining: 40.4ms
    3:	learn: 0.3149165	total: 30.6ms	remaining: 30.6ms
    4:	learn: 0.2730054	total: 37.7ms	remaining: 22.6ms
    5:	learn: 0.2480936	total: 44.2ms	remaining: 14.7ms
    6:	learn: 0.2242995	total: 50.6ms	remaining: 7.23ms
    7:	learn: 0.1956698	total: 59.8ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.4985520	total: 6.34ms	remaining: 44.4ms
    1:	learn: 0.4040345	total: 13.6ms	remaining: 40.8ms
    2:	learn: 0.2650671	total: 23.9ms	remaining: 39.9ms
    3:	learn: 0.2274985	total: 30.3ms	remaining: 30.3ms
    4:	learn: 0.2086369	total: 36.6ms	remaining: 22ms
    5:	learn: 0.1821157	total: 42.9ms	remaining: 14.3ms
    6:	learn: 0.1579086	total: 49.1ms	remaining: 7.02ms
    7:	learn: 0.1390924	total: 55.4ms	remaining: 0us
    Learning rate set to 0.208613
    0:	learn: 0.5498115	total: 6.51ms	remaining: 45.6ms
    1:	learn: 0.4273126	total: 12.8ms	remaining: 38.3ms
    2:	learn: 0.3261050	total: 18.9ms	remaining: 31.5ms
    3:	learn: 0.2864794	total: 25.3ms	remaining: 25.3ms
    4:	learn: 0.2566930	total: 31.8ms	remaining: 19.1ms
    5:	learn: 0.1662856	total: 38ms	remaining: 12.7ms
    6:	learn: 0.1473045	total: 44.3ms	remaining: 6.33ms
    7:	learn: 0.1313646	total: 50.5ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.5244093	total: 7.51ms	remaining: 60.1ms
    1:	learn: 0.4660514	total: 14.3ms	remaining: 50.1ms
    2:	learn: 0.4073267	total: 20.9ms	remaining: 41.7ms
    3:	learn: 0.3677037	total: 29.5ms	remaining: 36.9ms
    4:	learn: 0.3374210	total: 37.3ms	remaining: 29.9ms
    5:	learn: 0.2935517	total: 43.8ms	remaining: 21.9ms
    6:	learn: 0.2746552	total: 50.2ms	remaining: 14.3ms
    7:	learn: 0.2518276	total: 56.4ms	remaining: 7.05ms
    8:	learn: 0.2133333	total: 62.7ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.5875709	total: 6.32ms	remaining: 50.6ms
    1:	learn: 0.5237732	total: 13.3ms	remaining: 46.7ms
    2:	learn: 0.4414687	total: 19.6ms	remaining: 39.2ms
    3:	learn: 0.3994413	total: 25.9ms	remaining: 32.3ms
    4:	learn: 0.3405169	total: 32.3ms	remaining: 25.8ms
    5:	learn: 0.2873403	total: 38.7ms	remaining: 19.4ms
    6:	learn: 0.2431007	total: 45.2ms	remaining: 12.9ms
    7:	learn: 0.2180123	total: 51.8ms	remaining: 6.47ms
    8:	learn: 0.1773277	total: 58.2ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.5783593	total: 6.28ms	remaining: 50.2ms
    1:	learn: 0.4960110	total: 12.7ms	remaining: 44.4ms
    2:	learn: 0.4002423	total: 19.1ms	remaining: 38.1ms
    3:	learn: 0.3346484	total: 25.3ms	remaining: 31.6ms
    4:	learn: 0.2940280	total: 31.6ms	remaining: 25.3ms
    5:	learn: 0.2683839	total: 37.9ms	remaining: 19ms
    6:	learn: 0.2436646	total: 44.1ms	remaining: 12.6ms
    7:	learn: 0.2138744	total: 50.4ms	remaining: 6.3ms
    8:	learn: 0.1918464	total: 56.8ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.5150599	total: 6.54ms	remaining: 52.3ms
    1:	learn: 0.4237404	total: 12.6ms	remaining: 44.1ms
    2:	learn: 0.2876208	total: 19ms	remaining: 37.9ms
    3:	learn: 0.2480167	total: 25.3ms	remaining: 31.6ms
    4:	learn: 0.2285615	total: 31.6ms	remaining: 25.3ms
    5:	learn: 0.2016153	total: 37.9ms	remaining: 18.9ms
    6:	learn: 0.1773902	total: 44.3ms	remaining: 12.7ms
    7:	learn: 0.1570063	total: 50.7ms	remaining: 6.33ms
    8:	learn: 0.1408188	total: 57.1ms	remaining: 0us
    Learning rate set to 0.187256
    0:	learn: 0.5624144	total: 9.1ms	remaining: 72.8ms
    1:	learn: 0.4951177	total: 19.7ms	remaining: 68.8ms
    2:	learn: 0.3847437	total: 26.3ms	remaining: 52.5ms
    3:	learn: 0.3419193	total: 32.7ms	remaining: 40.9ms
    4:	learn: 0.3061038	total: 39.2ms	remaining: 31.4ms
    5:	learn: 0.2718936	total: 45.6ms	remaining: 22.8ms
    6:	learn: 0.2523161	total: 51.9ms	remaining: 14.8ms
    7:	learn: 0.2233694	total: 58.2ms	remaining: 7.28ms
    8:	learn: 0.1928037	total: 64.1ms	remaining: 0us
    Learning rate set to 0.336105
    0:	learn: 0.4973555	total: 14.1ms	remaining: 99ms
    1:	learn: 0.3952437	total: 28.4ms	remaining: 85.1ms
    2:	learn: 0.3494096	total: 42.1ms	remaining: 70.2ms
    3:	learn: 0.2863954	total: 59.8ms	remaining: 59.8ms
    4:	learn: 0.2560444	total: 73.8ms	remaining: 44.3ms
    5:	learn: 0.1997680	total: 87.6ms	remaining: 29.2ms
    6:	learn: 0.1807640	total: 101ms	remaining: 14.5ms
    7:	learn: 0.1705191	total: 115ms	remaining: 0us
    Learning rate set to 0.336105
    0:	learn: 0.5360274	total: 14.3ms	remaining: 100ms
    1:	learn: 0.3776226	total: 28.6ms	remaining: 85.9ms
    2:	learn: 0.3177258	total: 46.9ms	remaining: 78.2ms
    3:	learn: 0.2818732	total: 61.2ms	remaining: 61.2ms
    4:	learn: 0.2375320	total: 75.7ms	remaining: 45.4ms
    5:	learn: 0.2189781	total: 90ms	remaining: 30ms
    6:	learn: 0.1846588	total: 104ms	remaining: 14.9ms
    7:	learn: 0.1617767	total: 118ms	remaining: 0us
    Learning rate set to 0.336105
    0:	learn: 0.5639644	total: 14ms	remaining: 98ms
    1:	learn: 0.4844616	total: 28.3ms	remaining: 85ms
    2:	learn: 0.3373105	total: 42.3ms	remaining: 70.5ms
    3:	learn: 0.2759607	total: 56.5ms	remaining: 56.5ms
    4:	learn: 0.2442745	total: 71.3ms	remaining: 42.8ms
    5:	learn: 0.2166244	total: 85.4ms	remaining: 28.5ms
    6:	learn: 0.1825375	total: 99.7ms	remaining: 14.2ms
    7:	learn: 0.1589614	total: 115ms	remaining: 0us
    Learning rate set to 0.336105
    0:	learn: 0.4556626	total: 13.9ms	remaining: 97ms
    1:	learn: 0.3551138	total: 28ms	remaining: 84ms
    2:	learn: 0.2913745	total: 44.1ms	remaining: 73.5ms
    3:	learn: 0.2479053	total: 58.2ms	remaining: 58.2ms
    4:	learn: 0.2170672	total: 72.4ms	remaining: 43.4ms
    5:	learn: 0.1994854	total: 86.4ms	remaining: 28.8ms
    6:	learn: 0.1897889	total: 100ms	remaining: 14.3ms
    7:	learn: 0.1682093	total: 120ms	remaining: 0us
    Learning rate set to 0.336105
    0:	learn: 0.5266639	total: 13.7ms	remaining: 96.1ms
    1:	learn: 0.4331870	total: 27.5ms	remaining: 82.6ms
    2:	learn: 0.3986561	total: 41.4ms	remaining: 69ms
    3:	learn: 0.3443618	total: 55.1ms	remaining: 55.1ms
    4:	learn: 0.3065925	total: 68.9ms	remaining: 41.3ms
    5:	learn: 0.2793598	total: 82.8ms	remaining: 27.6ms
    6:	learn: 0.2239304	total: 96.7ms	remaining: 13.8ms
    7:	learn: 0.2188938	total: 112ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4917918	total: 7.06ms	remaining: 21.2ms
    1:	learn: 0.4081954	total: 14.3ms	remaining: 14.3ms
    2:	learn: 0.3474660	total: 21.3ms	remaining: 7.09ms
    3:	learn: 0.2784966	total: 28ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5257383	total: 13.4ms	remaining: 40.1ms
    1:	learn: 0.4386370	total: 20.9ms	remaining: 20.9ms
    2:	learn: 0.3432473	total: 28ms	remaining: 9.32ms
    3:	learn: 0.2982879	total: 34.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5426633	total: 8.23ms	remaining: 24.7ms
    1:	learn: 0.4404793	total: 15.4ms	remaining: 15.4ms
    2:	learn: 0.3790817	total: 22.6ms	remaining: 7.52ms
    3:	learn: 0.3516091	total: 29.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4807420	total: 6.91ms	remaining: 20.7ms
    1:	learn: 0.4176235	total: 17.7ms	remaining: 17.7ms
    2:	learn: 0.3823450	total: 24.8ms	remaining: 8.26ms
    3:	learn: 0.3337395	total: 31.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4899999	total: 8.59ms	remaining: 25.8ms
    1:	learn: 0.3993973	total: 15.1ms	remaining: 15.1ms
    2:	learn: 0.3268510	total: 26.2ms	remaining: 8.74ms
    3:	learn: 0.2896337	total: 33.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4398378	total: 14.1ms	remaining: 28.1ms
    1:	learn: 0.3479215	total: 30ms	remaining: 15ms
    2:	learn: 0.3074765	total: 47.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4905383	total: 13.9ms	remaining: 27.8ms
    1:	learn: 0.3279255	total: 28.4ms	remaining: 14.2ms
    2:	learn: 0.2567083	total: 43ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5232324	total: 14.5ms	remaining: 29ms
    1:	learn: 0.4025483	total: 29.4ms	remaining: 14.7ms
    2:	learn: 0.3545972	total: 44ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3933147	total: 14.2ms	remaining: 28.3ms
    1:	learn: 0.3002062	total: 28.5ms	remaining: 14.3ms
    2:	learn: 0.2482934	total: 42.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4750294	total: 17ms	remaining: 34ms
    1:	learn: 0.3846964	total: 31ms	remaining: 15.5ms
    2:	learn: 0.3196888	total: 44.7ms	remaining: 0us
    Learning rate set to 0.301695
    0:	learn: 0.5123712	total: 14ms	remaining: 112ms
    1:	learn: 0.4104750	total: 28.7ms	remaining: 101ms
    2:	learn: 0.3515564	total: 42.9ms	remaining: 85.8ms
    3:	learn: 0.3047994	total: 57ms	remaining: 71.2ms
    4:	learn: 0.2664071	total: 71ms	remaining: 56.8ms
    5:	learn: 0.2209324	total: 93.2ms	remaining: 46.6ms
    6:	learn: 0.2080922	total: 114ms	remaining: 32.6ms
    7:	learn: 0.1953807	total: 129ms	remaining: 16.1ms
    8:	learn: 0.1664663	total: 143ms	remaining: 0us
    Learning rate set to 0.301695
    0:	learn: 0.5479477	total: 13.8ms	remaining: 110ms
    1:	learn: 0.3933808	total: 29.7ms	remaining: 104ms
    2:	learn: 0.3352030	total: 43.7ms	remaining: 87.5ms
    3:	learn: 0.2900022	total: 57.4ms	remaining: 71.8ms
    4:	learn: 0.2287354	total: 71.4ms	remaining: 57.1ms
    5:	learn: 0.2077879	total: 85ms	remaining: 42.5ms
    6:	learn: 0.1954625	total: 98.9ms	remaining: 28.3ms
    7:	learn: 0.1725110	total: 113ms	remaining: 14.1ms
    8:	learn: 0.1562475	total: 127ms	remaining: 0us
    Learning rate set to 0.301695
    0:	learn: 0.5742735	total: 13.9ms	remaining: 111ms
    1:	learn: 0.4962290	total: 27.7ms	remaining: 97ms
    2:	learn: 0.3528550	total: 41.7ms	remaining: 83.5ms
    3:	learn: 0.2928182	total: 56ms	remaining: 70ms
    4:	learn: 0.2606964	total: 70.2ms	remaining: 56.2ms
    5:	learn: 0.2343330	total: 84.5ms	remaining: 42.2ms
    6:	learn: 0.2001209	total: 98.8ms	remaining: 28.2ms
    7:	learn: 0.1775918	total: 117ms	remaining: 14.7ms
    8:	learn: 0.1686173	total: 132ms	remaining: 0us
    Learning rate set to 0.301695
    0:	learn: 0.4726526	total: 14.9ms	remaining: 119ms
    1:	learn: 0.3759457	total: 28.7ms	remaining: 100ms
    2:	learn: 0.3371211	total: 42.5ms	remaining: 85.1ms
    3:	learn: 0.3027623	total: 65.1ms	remaining: 81.4ms
    4:	learn: 0.2489824	total: 83.1ms	remaining: 66.4ms
    5:	learn: 0.2175225	total: 97.2ms	remaining: 48.6ms
    6:	learn: 0.2066571	total: 111ms	remaining: 31.7ms
    7:	learn: 0.1983125	total: 125ms	remaining: 15.6ms
    8:	learn: 0.1845994	total: 139ms	remaining: 0us
    Learning rate set to 0.301695
    0:	learn: 0.5398420	total: 13.9ms	remaining: 111ms
    1:	learn: 0.4472927	total: 27.5ms	remaining: 96.1ms
    2:	learn: 0.4099715	total: 40.7ms	remaining: 81.4ms
    3:	learn: 0.3564952	total: 54.2ms	remaining: 67.8ms
    4:	learn: 0.3261851	total: 68.7ms	remaining: 54.9ms
    5:	learn: 0.2985048	total: 82.3ms	remaining: 41.2ms
    6:	learn: 0.2642459	total: 96.2ms	remaining: 27.5ms
    7:	learn: 0.2457270	total: 110ms	remaining: 13.7ms
    8:	learn: 0.2203659	total: 124ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4398378	total: 14.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4905383	total: 13.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5232324	total: 14.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3933147	total: 19.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4750294	total: 14.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4398378	total: 13.9ms	remaining: 41.7ms
    1:	learn: 0.3479215	total: 28.6ms	remaining: 28.6ms
    2:	learn: 0.3074765	total: 43.2ms	remaining: 14.4ms
    3:	learn: 0.2459370	total: 59.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4905383	total: 14.2ms	remaining: 42.5ms
    1:	learn: 0.3279255	total: 28.2ms	remaining: 28.2ms
    2:	learn: 0.2567083	total: 42.6ms	remaining: 14.2ms
    3:	learn: 0.2146734	total: 56.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5232324	total: 14.4ms	remaining: 43.3ms
    1:	learn: 0.4025483	total: 29ms	remaining: 29ms
    2:	learn: 0.3545972	total: 46.8ms	remaining: 15.6ms
    3:	learn: 0.3137516	total: 76ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.3933147	total: 14ms	remaining: 41.9ms
    1:	learn: 0.3002062	total: 30ms	remaining: 30ms
    2:	learn: 0.2482934	total: 43.9ms	remaining: 14.6ms
    3:	learn: 0.2146098	total: 58.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4750294	total: 13.9ms	remaining: 41.7ms
    1:	learn: 0.3846964	total: 28.2ms	remaining: 28.2ms
    2:	learn: 0.3196888	total: 42.1ms	remaining: 14ms
    3:	learn: 0.2866522	total: 56.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5193690	total: 29.3ms	remaining: 87.9ms
    1:	learn: 0.4245245	total: 58.9ms	remaining: 58.9ms
    2:	learn: 0.3275531	total: 89.1ms	remaining: 29.7ms
    3:	learn: 0.3031857	total: 119ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5029144	total: 29.5ms	remaining: 88.5ms
    1:	learn: 0.4293650	total: 59.7ms	remaining: 59.7ms
    2:	learn: 0.3687041	total: 89.3ms	remaining: 29.8ms
    3:	learn: 0.3291995	total: 119ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4984925	total: 29.3ms	remaining: 88ms
    1:	learn: 0.3935633	total: 59ms	remaining: 59ms
    2:	learn: 0.3415986	total: 92.5ms	remaining: 30.8ms
    3:	learn: 0.2869272	total: 123ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5103648	total: 32.4ms	remaining: 97.1ms
    1:	learn: 0.4179493	total: 62ms	remaining: 62ms
    2:	learn: 0.3654590	total: 91.5ms	remaining: 30.5ms
    3:	learn: 0.3129787	total: 126ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4937285	total: 29.2ms	remaining: 87.6ms
    1:	learn: 0.4061917	total: 58.7ms	remaining: 58.7ms
    2:	learn: 0.3430064	total: 88.7ms	remaining: 29.6ms
    3:	learn: 0.3029805	total: 120ms	remaining: 0us
    Learning rate set to 0.482903
    0:	learn: 0.5229306	total: 29.4ms	remaining: 236ms
    1:	learn: 0.4276606	total: 59ms	remaining: 207ms
    2:	learn: 0.3311793	total: 89ms	remaining: 178ms
    3:	learn: 0.3068440	total: 119ms	remaining: 148ms
    4:	learn: 0.2862506	total: 148ms	remaining: 118ms
    5:	learn: 0.2646463	total: 178ms	remaining: 89.2ms
    6:	learn: 0.2458521	total: 212ms	remaining: 60.7ms
    7:	learn: 0.2281588	total: 243ms	remaining: 30.4ms
    8:	learn: 0.2016100	total: 273ms	remaining: 0us
    Learning rate set to 0.482903
    0:	learn: 0.5067355	total: 29.8ms	remaining: 239ms
    1:	learn: 0.4322434	total: 59.3ms	remaining: 208ms
    2:	learn: 0.3646562	total: 88.9ms	remaining: 178ms
    3:	learn: 0.3132760	total: 119ms	remaining: 148ms
    4:	learn: 0.2851927	total: 155ms	remaining: 124ms
    5:	learn: 0.2346989	total: 186ms	remaining: 93ms
    6:	learn: 0.2275828	total: 222ms	remaining: 63.5ms
    7:	learn: 0.1930569	total: 252ms	remaining: 31.5ms
    8:	learn: 0.1771595	total: 284ms	remaining: 0us
    Learning rate set to 0.482903
    0:	learn: 0.5024613	total: 29.4ms	remaining: 235ms
    1:	learn: 0.3971660	total: 63.1ms	remaining: 221ms
    2:	learn: 0.3453785	total: 91.9ms	remaining: 184ms
    3:	learn: 0.2910319	total: 121ms	remaining: 152ms
    4:	learn: 0.2588848	total: 154ms	remaining: 123ms
    5:	learn: 0.2228396	total: 183ms	remaining: 91.6ms
    6:	learn: 0.2140385	total: 217ms	remaining: 62ms
    7:	learn: 0.2008189	total: 247ms	remaining: 30.9ms
    8:	learn: 0.1795681	total: 276ms	remaining: 0us
    Learning rate set to 0.482903
    0:	learn: 0.5140607	total: 29.2ms	remaining: 234ms
    1:	learn: 0.4210939	total: 58.5ms	remaining: 205ms
    2:	learn: 0.3683037	total: 90ms	remaining: 180ms
    3:	learn: 0.3227801	total: 119ms	remaining: 149ms
    4:	learn: 0.2866767	total: 150ms	remaining: 120ms
    5:	learn: 0.2460059	total: 184ms	remaining: 92.1ms
    6:	learn: 0.2271555	total: 218ms	remaining: 62.4ms
    7:	learn: 0.2063240	total: 249ms	remaining: 31.1ms
    8:	learn: 0.1783018	total: 278ms	remaining: 0us
    Learning rate set to 0.482903
    0:	learn: 0.4977736	total: 30.8ms	remaining: 247ms
    1:	learn: 0.4012287	total: 60.8ms	remaining: 213ms
    2:	learn: 0.3538782	total: 91.2ms	remaining: 182ms
    3:	learn: 0.3121752	total: 121ms	remaining: 152ms
    4:	learn: 0.2784082	total: 152ms	remaining: 121ms
    5:	learn: 0.2472360	total: 182ms	remaining: 91.1ms
    6:	learn: 0.2335564	total: 217ms	remaining: 61.9ms
    7:	learn: 0.2111509	total: 247ms	remaining: 30.9ms
    8:	learn: 0.1977754	total: 277ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5160889	total: 29.2ms	remaining: 234ms
    1:	learn: 0.4423533	total: 63.9ms	remaining: 224ms
    2:	learn: 0.4042982	total: 94.1ms	remaining: 188ms
    3:	learn: 0.3582334	total: 124ms	remaining: 155ms
    4:	learn: 0.3040232	total: 154ms	remaining: 123ms
    5:	learn: 0.2723878	total: 183ms	remaining: 91.5ms
    6:	learn: 0.2584721	total: 218ms	remaining: 62.2ms
    7:	learn: 0.2382580	total: 248ms	remaining: 31ms
    8:	learn: 0.2198965	total: 277ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5383656	total: 5.86ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5607593	total: 3.32ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4385891	total: 3.31ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5383656	total: 3.41ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5703858	total: 3.44ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5383656	total: 3.37ms	remaining: 3.37ms
    1:	learn: 0.3109683	total: 7.05ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5607593	total: 3.33ms	remaining: 3.33ms
    1:	learn: 0.4666109	total: 7.03ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4385891	total: 3.37ms	remaining: 3.37ms
    1:	learn: 0.3235166	total: 7.06ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5383656	total: 3.42ms	remaining: 3.42ms
    1:	learn: 0.4498094	total: 7.29ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5703858	total: 3.3ms	remaining: 3.3ms
    1:	learn: 0.3311801	total: 6.87ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5383656	total: 3.43ms	remaining: 6.85ms
    1:	learn: 0.3109683	total: 7.61ms	remaining: 3.81ms
    2:	learn: 0.2468032	total: 13.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5607593	total: 3.41ms	remaining: 6.82ms
    1:	learn: 0.4666109	total: 7.26ms	remaining: 3.63ms
    2:	learn: 0.3796374	total: 10.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4385891	total: 3.38ms	remaining: 6.77ms
    1:	learn: 0.3235166	total: 6.99ms	remaining: 3.5ms
    2:	learn: 0.2563556	total: 10.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5383656	total: 3.38ms	remaining: 6.75ms
    1:	learn: 0.4498094	total: 7.01ms	remaining: 3.5ms
    2:	learn: 0.3545989	total: 10.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5703858	total: 3.29ms	remaining: 6.57ms
    1:	learn: 0.3311801	total: 6.71ms	remaining: 3.35ms
    2:	learn: 0.2408380	total: 9.99ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.5568376	total: 3.59ms	remaining: 10.8ms
    1:	learn: 0.4653028	total: 7.42ms	remaining: 7.42ms
    2:	learn: 0.3714187	total: 11.1ms	remaining: 3.69ms
    3:	learn: 0.2636899	total: 14.7ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.5766066	total: 3.46ms	remaining: 10.4ms
    1:	learn: 0.4807598	total: 7.16ms	remaining: 7.16ms
    2:	learn: 0.3937503	total: 14ms	remaining: 4.68ms
    3:	learn: 0.3100431	total: 19.9ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.4659063	total: 3.47ms	remaining: 10.4ms
    1:	learn: 0.3577296	total: 7.17ms	remaining: 7.17ms
    2:	learn: 0.2894523	total: 10.6ms	remaining: 3.53ms
    3:	learn: 0.2664110	total: 14.3ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.5568376	total: 3.37ms	remaining: 10.1ms
    1:	learn: 0.4441215	total: 6.95ms	remaining: 6.95ms
    2:	learn: 0.3489479	total: 15.3ms	remaining: 5.11ms
    3:	learn: 0.3017324	total: 18.6ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.5853310	total: 3.26ms	remaining: 9.77ms
    1:	learn: 0.3616930	total: 6.82ms	remaining: 6.82ms
    2:	learn: 0.2753546	total: 10.3ms	remaining: 3.44ms
    3:	learn: 0.2073647	total: 13.4ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.5757240	total: 3.31ms	remaining: 13.3ms
    1:	learn: 0.4847722	total: 6.81ms	remaining: 10.2ms
    2:	learn: 0.3929799	total: 10.2ms	remaining: 6.78ms
    3:	learn: 0.2940805	total: 13.6ms	remaining: 3.41ms
    4:	learn: 0.2632954	total: 17.2ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.5927916	total: 3.31ms	remaining: 13.2ms
    1:	learn: 0.5365415	total: 6.85ms	remaining: 10.3ms
    2:	learn: 0.4310739	total: 10.4ms	remaining: 6.97ms
    3:	learn: 0.3550936	total: 14.3ms	remaining: 3.57ms
    4:	learn: 0.3385842	total: 18.1ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.4950387	total: 3.36ms	remaining: 13.4ms
    1:	learn: 0.3945989	total: 7ms	remaining: 10.5ms
    2:	learn: 0.3257214	total: 10.6ms	remaining: 7.08ms
    3:	learn: 0.3010702	total: 14.4ms	remaining: 3.59ms
    4:	learn: 0.2774943	total: 17.8ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.5757241	total: 3.4ms	remaining: 13.6ms
    1:	learn: 0.5208402	total: 7.15ms	remaining: 10.7ms
    2:	learn: 0.4027200	total: 10.7ms	remaining: 7.13ms
    3:	learn: 0.3504018	total: 14.3ms	remaining: 3.58ms
    4:	learn: 0.2889277	total: 17.9ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.6005027	total: 3.3ms	remaining: 13.2ms
    1:	learn: 0.3963048	total: 6.77ms	remaining: 10.2ms
    2:	learn: 0.3134886	total: 10ms	remaining: 6.69ms
    3:	learn: 0.2422013	total: 13.4ms	remaining: 3.35ms
    4:	learn: 0.1982795	total: 16.8ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5901145	total: 3.34ms	remaining: 16.7ms
    1:	learn: 0.5021382	total: 6.93ms	remaining: 13.9ms
    2:	learn: 0.4123583	total: 10.4ms	remaining: 10.4ms
    3:	learn: 0.3193008	total: 13.9ms	remaining: 6.95ms
    4:	learn: 0.2903194	total: 17.3ms	remaining: 3.47ms
    5:	learn: 0.2576832	total: 20.7ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.6051138	total: 5.83ms	remaining: 29.1ms
    1:	learn: 0.5495114	total: 9.87ms	remaining: 19.7ms
    2:	learn: 0.4491867	total: 13.6ms	remaining: 13.6ms
    3:	learn: 0.3779994	total: 17.2ms	remaining: 8.58ms
    4:	learn: 0.3621287	total: 20.7ms	remaining: 4.14ms
    5:	learn: 0.3405250	total: 25.3ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5178917	total: 3.38ms	remaining: 16.9ms
    1:	learn: 0.4243436	total: 7.08ms	remaining: 14.2ms
    2:	learn: 0.3558614	total: 10.8ms	remaining: 10.8ms
    3:	learn: 0.3300943	total: 14.4ms	remaining: 7.22ms
    4:	learn: 0.3051883	total: 18ms	remaining: 3.59ms
    5:	learn: 0.2824970	total: 23.6ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5901145	total: 3.4ms	remaining: 17ms
    1:	learn: 0.5349781	total: 7.04ms	remaining: 14.1ms
    2:	learn: 0.4239300	total: 10.6ms	remaining: 10.6ms
    3:	learn: 0.3708921	total: 14.2ms	remaining: 7.09ms
    4:	learn: 0.3142274	total: 17.7ms	remaining: 3.55ms
    5:	learn: 0.2812969	total: 21.2ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.6119979	total: 3.3ms	remaining: 16.5ms
    1:	learn: 0.4250554	total: 6.83ms	remaining: 13.7ms
    2:	learn: 0.3452808	total: 10.3ms	remaining: 10.3ms
    3:	learn: 0.2721727	total: 13.6ms	remaining: 6.78ms
    4:	learn: 0.2258846	total: 16.9ms	remaining: 3.39ms
    5:	learn: 0.2047563	total: 20.3ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.6013524	total: 3.31ms	remaining: 19.8ms
    1:	learn: 0.5172486	total: 6.98ms	remaining: 17.5ms
    2:	learn: 0.4297867	total: 10.7ms	remaining: 14.3ms
    3:	learn: 0.3410444	total: 14.2ms	remaining: 10.7ms
    4:	learn: 0.3129210	total: 17.7ms	remaining: 7.08ms
    5:	learn: 0.2813727	total: 21.4ms	remaining: 3.56ms
    6:	learn: 0.2266218	total: 24.8ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.6147330	total: 3.37ms	remaining: 20.2ms
    1:	learn: 0.5608537	total: 6.95ms	remaining: 17.4ms
    2:	learn: 0.4653938	total: 10.4ms	remaining: 13.8ms
    3:	learn: 0.3976308	total: 13.9ms	remaining: 10.4ms
    4:	learn: 0.3817949	total: 17.3ms	remaining: 6.92ms
    5:	learn: 0.3610870	total: 20.8ms	remaining: 3.47ms
    6:	learn: 0.3197979	total: 24.3ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.5360932	total: 3.43ms	remaining: 20.6ms
    1:	learn: 0.4487031	total: 7.03ms	remaining: 17.6ms
    2:	learn: 0.3813637	total: 10.6ms	remaining: 14.1ms
    3:	learn: 0.3548361	total: 14.3ms	remaining: 10.7ms
    4:	learn: 0.3290242	total: 18.1ms	remaining: 7.23ms
    5:	learn: 0.3066765	total: 21.7ms	remaining: 3.62ms
    6:	learn: 0.2794812	total: 29.2ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.6013524	total: 3.54ms	remaining: 21.2ms
    1:	learn: 0.5473627	total: 9.87ms	remaining: 24.7ms
    2:	learn: 0.4424833	total: 13.8ms	remaining: 18.5ms
    3:	learn: 0.3891250	total: 17.2ms	remaining: 12.9ms
    4:	learn: 0.3356216	total: 20.7ms	remaining: 8.27ms
    5:	learn: 0.3019657	total: 24.1ms	remaining: 4.02ms
    6:	learn: 0.2729895	total: 27.5ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.6209414	total: 3.34ms	remaining: 20ms
    1:	learn: 0.4489709	total: 7ms	remaining: 17.5ms
    2:	learn: 0.3812850	total: 10.5ms	remaining: 14ms
    3:	learn: 0.3265118	total: 13.9ms	remaining: 10.4ms
    4:	learn: 0.2680861	total: 17.3ms	remaining: 6.92ms
    5:	learn: 0.2482346	total: 20.8ms	remaining: 3.47ms
    6:	learn: 0.2291011	total: 24.4ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.6103516	total: 3.57ms	remaining: 25ms
    1:	learn: 0.5303424	total: 6.96ms	remaining: 20.9ms
    2:	learn: 0.4454624	total: 10.4ms	remaining: 17.3ms
    3:	learn: 0.3602081	total: 13.7ms	remaining: 13.7ms
    4:	learn: 0.3324361	total: 17ms	remaining: 10.2ms
    5:	learn: 0.3015566	total: 20.3ms	remaining: 6.78ms
    6:	learn: 0.2476532	total: 23.6ms	remaining: 3.36ms
    7:	learn: 0.2335940	total: 26.9ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.6224277	total: 3.37ms	remaining: 23.6ms
    1:	learn: 0.5706852	total: 7.2ms	remaining: 21.6ms
    2:	learn: 0.4798112	total: 10.7ms	remaining: 17.9ms
    3:	learn: 0.4147963	total: 14.4ms	remaining: 14.4ms
    4:	learn: 0.3987237	total: 18.1ms	remaining: 10.8ms
    5:	learn: 0.3784719	total: 21.8ms	remaining: 7.26ms
    6:	learn: 0.3395356	total: 30ms	remaining: 4.29ms
    7:	learn: 0.3034071	total: 34.7ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.5508570	total: 3.4ms	remaining: 23.8ms
    1:	learn: 0.4689826	total: 7.11ms	remaining: 21.3ms
    2:	learn: 0.4032146	total: 10.7ms	remaining: 17.8ms
    3:	learn: 0.3762431	total: 14.4ms	remaining: 14.4ms
    4:	learn: 0.3498351	total: 17.9ms	remaining: 10.7ms
    5:	learn: 0.3283351	total: 21.4ms	remaining: 7.14ms
    6:	learn: 0.3015251	total: 25ms	remaining: 3.57ms
    7:	learn: 0.2659385	total: 32.5ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.6103516	total: 3.46ms	remaining: 24.3ms
    1:	learn: 0.5581131	total: 7.45ms	remaining: 22.3ms
    2:	learn: 0.4587862	total: 11.2ms	remaining: 18.7ms
    3:	learn: 0.4055095	total: 15ms	remaining: 15ms
    4:	learn: 0.3542357	total: 18.6ms	remaining: 11.2ms
    5:	learn: 0.3199960	total: 22.3ms	remaining: 7.42ms
    6:	learn: 0.2911459	total: 25.8ms	remaining: 3.69ms
    7:	learn: 0.2695470	total: 29.5ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.6280852	total: 3.31ms	remaining: 23.1ms
    1:	learn: 0.4690404	total: 6.69ms	remaining: 20.1ms
    2:	learn: 0.4034706	total: 10ms	remaining: 16.7ms
    3:	learn: 0.3477973	total: 13.3ms	remaining: 13.3ms
    4:	learn: 0.2890864	total: 16.8ms	remaining: 10.1ms
    5:	learn: 0.2685103	total: 20.2ms	remaining: 6.73ms
    6:	learn: 0.2493346	total: 23.5ms	remaining: 3.36ms
    7:	learn: 0.2418879	total: 26.9ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.6177044	total: 3.32ms	remaining: 26.6ms
    1:	learn: 0.5417079	total: 6.9ms	remaining: 24.2ms
    2:	learn: 0.4595475	total: 10.4ms	remaining: 20.8ms
    3:	learn: 0.3773272	total: 13.8ms	remaining: 17.3ms
    4:	learn: 0.3496684	total: 17.4ms	remaining: 13.9ms
    5:	learn: 0.3191932	total: 20.8ms	remaining: 10.4ms
    6:	learn: 0.2662623	total: 24.2ms	remaining: 6.92ms
    7:	learn: 0.2520897	total: 27.6ms	remaining: 3.45ms
    8:	learn: 0.2160501	total: 32.4ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.6287168	total: 3.4ms	remaining: 27.2ms
    1:	learn: 0.5792335	total: 7.06ms	remaining: 24.7ms
    2:	learn: 0.4926481	total: 11.4ms	remaining: 22.7ms
    3:	learn: 0.4300172	total: 14.8ms	remaining: 18.5ms
    4:	learn: 0.4136183	total: 18.3ms	remaining: 14.6ms
    5:	learn: 0.3935856	total: 21.8ms	remaining: 10.9ms
    6:	learn: 0.3564857	total: 25.3ms	remaining: 7.22ms
    7:	learn: 0.3213394	total: 28.7ms	remaining: 3.59ms
    8:	learn: 0.3115396	total: 32.2ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.5630443	total: 3.81ms	remaining: 30.4ms
    1:	learn: 0.4860956	total: 7.32ms	remaining: 25.6ms
    2:	learn: 0.4221632	total: 10.5ms	remaining: 21.1ms
    3:	learn: 0.3949791	total: 13.7ms	remaining: 17.1ms
    4:	learn: 0.3682572	total: 17.1ms	remaining: 13.7ms
    5:	learn: 0.3473756	total: 20.5ms	remaining: 10.3ms
    6:	learn: 0.3210155	total: 24.1ms	remaining: 6.88ms
    7:	learn: 0.2854548	total: 27.6ms	remaining: 3.44ms
    8:	learn: 0.2701456	total: 31.3ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.6177044	total: 3.42ms	remaining: 27.3ms
    1:	learn: 0.5674658	total: 7.28ms	remaining: 25.5ms
    2:	learn: 0.4731740	total: 12.1ms	remaining: 24.1ms
    3:	learn: 0.4202982	total: 22.9ms	remaining: 28.7ms
    4:	learn: 0.3707429	total: 28.9ms	remaining: 23.1ms
    5:	learn: 0.3360442	total: 35.1ms	remaining: 17.5ms
    6:	learn: 0.3073329	total: 38.5ms	remaining: 11ms
    7:	learn: 0.2861703	total: 41.8ms	remaining: 5.22ms
    8:	learn: 0.2573426	total: 45.1ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.6339105	total: 3.34ms	remaining: 26.7ms
    1:	learn: 0.4860776	total: 6.46ms	remaining: 22.6ms
    2:	learn: 0.4226577	total: 9.98ms	remaining: 20ms
    3:	learn: 0.3667331	total: 14ms	remaining: 17.5ms
    4:	learn: 0.3080502	total: 17.4ms	remaining: 13.9ms
    5:	learn: 0.2868250	total: 20.8ms	remaining: 10.4ms
    6:	learn: 0.2675837	total: 24.2ms	remaining: 6.9ms
    7:	learn: 0.2055527	total: 27.6ms	remaining: 3.44ms
    8:	learn: 0.1937809	total: 30.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4718777	total: 6.81ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5013595	total: 6.68ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4283926	total: 6.68ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4195960	total: 6.74ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4903825	total: 6.94ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4718777	total: 6.82ms	remaining: 6.82ms
    1:	learn: 0.3302599	total: 13.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5013595	total: 6.7ms	remaining: 6.7ms
    1:	learn: 0.3993685	total: 13.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4283926	total: 6.47ms	remaining: 6.47ms
    1:	learn: 0.3210167	total: 13.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4195960	total: 6.89ms	remaining: 6.89ms
    1:	learn: 0.2893766	total: 13.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4903825	total: 6.53ms	remaining: 6.53ms
    1:	learn: 0.2721006	total: 13ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4718777	total: 8.65ms	remaining: 17.3ms
    1:	learn: 0.3302599	total: 15.2ms	remaining: 7.62ms
    2:	learn: 0.2026319	total: 21.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5013595	total: 6.58ms	remaining: 13.2ms
    1:	learn: 0.3993685	total: 13.4ms	remaining: 6.7ms
    2:	learn: 0.2934870	total: 20.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4283926	total: 6.72ms	remaining: 13.4ms
    1:	learn: 0.3210167	total: 13.6ms	remaining: 6.78ms
    2:	learn: 0.1844986	total: 20.3ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4195960	total: 6.58ms	remaining: 13.2ms
    1:	learn: 0.2893766	total: 13.2ms	remaining: 6.61ms
    2:	learn: 0.2189983	total: 19.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4903825	total: 6.43ms	remaining: 12.9ms
    1:	learn: 0.2721006	total: 12.8ms	remaining: 6.42ms
    2:	learn: 0.1590864	total: 19.2ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.4960481	total: 6.64ms	remaining: 19.9ms
    1:	learn: 0.3591971	total: 13.5ms	remaining: 13.5ms
    2:	learn: 0.2305713	total: 22.3ms	remaining: 7.43ms
    3:	learn: 0.1819968	total: 31ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.5237328	total: 6.75ms	remaining: 20.3ms
    1:	learn: 0.4260111	total: 13.5ms	remaining: 13.5ms
    2:	learn: 0.3528861	total: 20.3ms	remaining: 6.76ms
    3:	learn: 0.2815944	total: 27.2ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.4582353	total: 6.47ms	remaining: 19.4ms
    1:	learn: 0.3534208	total: 13.2ms	remaining: 13.2ms
    2:	learn: 0.2160604	total: 20.1ms	remaining: 6.71ms
    3:	learn: 0.1606979	total: 26.8ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.4503936	total: 6.88ms	remaining: 20.7ms
    1:	learn: 0.3271492	total: 13.8ms	remaining: 13.8ms
    2:	learn: 0.2266685	total: 20.5ms	remaining: 6.82ms
    3:	learn: 0.1567890	total: 27.4ms	remaining: 0us
    Learning rate set to 0.412026
    0:	learn: 0.5134755	total: 6.54ms	remaining: 19.6ms
    1:	learn: 0.3077261	total: 13.2ms	remaining: 13.2ms
    2:	learn: 0.1939184	total: 27.2ms	remaining: 9.06ms
    3:	learn: 0.1477079	total: 33.5ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.5213727	total: 6.58ms	remaining: 26.3ms
    1:	learn: 0.3912138	total: 13.5ms	remaining: 20.2ms
    2:	learn: 0.2632733	total: 20.1ms	remaining: 13.4ms
    3:	learn: 0.1912228	total: 26.7ms	remaining: 6.67ms
    4:	learn: 0.1549915	total: 33.2ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.5466348	total: 6.61ms	remaining: 26.4ms
    1:	learn: 0.4546383	total: 13.4ms	remaining: 20.1ms
    2:	learn: 0.3788124	total: 20.1ms	remaining: 13.4ms
    3:	learn: 0.3124058	total: 26.6ms	remaining: 6.66ms
    4:	learn: 0.2592577	total: 33.2ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.4894086	total: 6.72ms	remaining: 26.9ms
    1:	learn: 0.3890124	total: 13.9ms	remaining: 20.8ms
    2:	learn: 0.2529872	total: 20.8ms	remaining: 13.9ms
    3:	learn: 0.1865743	total: 32.7ms	remaining: 8.19ms
    4:	learn: 0.1634233	total: 41.2ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.4824850	total: 6.72ms	remaining: 26.9ms
    1:	learn: 0.4332026	total: 13.9ms	remaining: 20.8ms
    2:	learn: 0.3502449	total: 20.6ms	remaining: 13.7ms
    3:	learn: 0.2298875	total: 27.5ms	remaining: 6.86ms
    4:	learn: 0.2027839	total: 34.2ms	remaining: 0us
    Learning rate set to 0.335783
    0:	learn: 0.5375087	total: 6.79ms	remaining: 27.2ms
    1:	learn: 0.3473736	total: 13.9ms	remaining: 20.8ms
    2:	learn: 0.2635826	total: 20.7ms	remaining: 13.8ms
    3:	learn: 0.1867872	total: 29.6ms	remaining: 7.39ms
    4:	learn: 0.1581493	total: 36ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5411281	total: 6.66ms	remaining: 33.3ms
    1:	learn: 0.4179045	total: 16.9ms	remaining: 33.7ms
    2:	learn: 0.2920779	total: 29ms	remaining: 29ms
    3:	learn: 0.2157513	total: 36.2ms	remaining: 18.1ms
    4:	learn: 0.1814475	total: 42.8ms	remaining: 8.57ms
    5:	learn: 0.1493084	total: 49.5ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5641747	total: 6.63ms	remaining: 33.2ms
    1:	learn: 0.4778335	total: 13.5ms	remaining: 27ms
    2:	learn: 0.4014351	total: 20.1ms	remaining: 20.1ms
    3:	learn: 0.3380116	total: 26.8ms	remaining: 13.4ms
    4:	learn: 0.2666558	total: 40ms	remaining: 8ms
    5:	learn: 0.2305345	total: 47.5ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5135031	total: 6.87ms	remaining: 34.4ms
    1:	learn: 0.4181799	total: 18.9ms	remaining: 37.8ms
    2:	learn: 0.2852912	total: 26.6ms	remaining: 26.6ms
    3:	learn: 0.2170711	total: 33.3ms	remaining: 16.7ms
    4:	learn: 0.1697040	total: 43.3ms	remaining: 8.65ms
    5:	learn: 0.1498615	total: 53.4ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5073335	total: 6.8ms	remaining: 34ms
    1:	learn: 0.4560560	total: 13.6ms	remaining: 27.2ms
    2:	learn: 0.3744263	total: 20.5ms	remaining: 20.5ms
    3:	learn: 0.2584689	total: 27.3ms	remaining: 13.7ms
    4:	learn: 0.2289969	total: 34ms	remaining: 6.81ms
    5:	learn: 0.1889697	total: 50.5ms	remaining: 0us
    Learning rate set to 0.284085
    0:	learn: 0.5560535	total: 6.52ms	remaining: 32.6ms
    1:	learn: 0.3801653	total: 13.3ms	remaining: 26.5ms
    2:	learn: 0.2928577	total: 19.9ms	remaining: 19.9ms
    3:	learn: 0.2289703	total: 26.5ms	remaining: 13.2ms
    4:	learn: 0.1946229	total: 35.8ms	remaining: 7.17ms
    5:	learn: 0.1270846	total: 42.5ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.5568380	total: 6.64ms	remaining: 39.8ms
    1:	learn: 0.4404097	total: 13.4ms	remaining: 33.6ms
    2:	learn: 0.3175757	total: 20.1ms	remaining: 26.8ms
    3:	learn: 0.2383933	total: 26.9ms	remaining: 20.2ms
    4:	learn: 0.2052595	total: 33.6ms	remaining: 13.4ms
    5:	learn: 0.1708357	total: 40.2ms	remaining: 6.7ms
    6:	learn: 0.1421458	total: 46.9ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.5779524	total: 9.42ms	remaining: 56.5ms
    1:	learn: 0.4969395	total: 18.3ms	remaining: 45.7ms
    2:	learn: 0.4212595	total: 24.7ms	remaining: 32.9ms
    3:	learn: 0.3599974	total: 31ms	remaining: 23.3ms
    4:	learn: 0.2911488	total: 37.4ms	remaining: 15ms
    5:	learn: 0.2561922	total: 44.4ms	remaining: 7.39ms
    6:	learn: 0.2024695	total: 51.3ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.5325283	total: 6.62ms	remaining: 39.7ms
    1:	learn: 0.4423460	total: 17.3ms	remaining: 43.3ms
    2:	learn: 0.3135751	total: 25.1ms	remaining: 33.5ms
    3:	learn: 0.2443498	total: 31.6ms	remaining: 23.7ms
    4:	learn: 0.1960853	total: 39ms	remaining: 15.6ms
    5:	learn: 0.1752769	total: 45.8ms	remaining: 7.63ms
    6:	learn: 0.1612209	total: 52.5ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.5269576	total: 6.68ms	remaining: 40.1ms
    1:	learn: 0.4755034	total: 13.7ms	remaining: 34.2ms
    2:	learn: 0.3956633	total: 21.1ms	remaining: 28.1ms
    3:	learn: 0.2836316	total: 28.7ms	remaining: 21.5ms
    4:	learn: 0.2522429	total: 36.2ms	remaining: 14.5ms
    5:	learn: 0.2263168	total: 43.2ms	remaining: 7.2ms
    6:	learn: 0.1991803	total: 50.1ms	remaining: 0us
    Learning rate set to 0.246637
    0:	learn: 0.5706657	total: 8.68ms	remaining: 52.1ms
    1:	learn: 0.4074573	total: 15.4ms	remaining: 38.4ms
    2:	learn: 0.3187763	total: 22.2ms	remaining: 29.6ms
    3:	learn: 0.2539387	total: 28.9ms	remaining: 21.7ms
    4:	learn: 0.2197321	total: 35.7ms	remaining: 14.3ms
    5:	learn: 0.1499397	total: 42.5ms	remaining: 7.08ms
    6:	learn: 0.1049876	total: 49.1ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.5695829	total: 7.63ms	remaining: 53.4ms
    1:	learn: 0.4595783	total: 20.2ms	remaining: 60.5ms
    2:	learn: 0.3402261	total: 26.9ms	remaining: 44.8ms
    3:	learn: 0.2593147	total: 33.7ms	remaining: 33.7ms
    4:	learn: 0.2231054	total: 41.1ms	remaining: 24.7ms
    5:	learn: 0.1892751	total: 47.9ms	remaining: 16ms
    6:	learn: 0.1606213	total: 54.7ms	remaining: 7.82ms
    7:	learn: 0.1339116	total: 61.3ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.5890320	total: 11.5ms	remaining: 80.6ms
    1:	learn: 0.5129222	total: 18.9ms	remaining: 56.7ms
    2:	learn: 0.4387133	total: 26.9ms	remaining: 44.9ms
    3:	learn: 0.3792485	total: 33.9ms	remaining: 33.9ms
    4:	learn: 0.3287538	total: 40.6ms	remaining: 24.4ms
    5:	learn: 0.2960201	total: 47.6ms	remaining: 15.9ms
    6:	learn: 0.2333643	total: 54.4ms	remaining: 7.76ms
    7:	learn: 0.1970278	total: 61.1ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.5478740	total: 11ms	remaining: 76.9ms
    1:	learn: 0.4626318	total: 17.6ms	remaining: 52.9ms
    2:	learn: 0.3384106	total: 24.3ms	remaining: 40.5ms
    3:	learn: 0.2688300	total: 31ms	remaining: 31ms
    4:	learn: 0.2430764	total: 37.9ms	remaining: 22.7ms
    5:	learn: 0.2118528	total: 44.6ms	remaining: 14.9ms
    6:	learn: 0.1937550	total: 51.5ms	remaining: 7.36ms
    7:	learn: 0.1825791	total: 58.5ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.5427945	total: 9.01ms	remaining: 63.1ms
    1:	learn: 0.4921194	total: 16.4ms	remaining: 49.1ms
    2:	learn: 0.4144019	total: 23.8ms	remaining: 39.7ms
    3:	learn: 0.3060395	total: 30.9ms	remaining: 30.9ms
    4:	learn: 0.2731281	total: 38.1ms	remaining: 22.8ms
    5:	learn: 0.2466379	total: 45.1ms	remaining: 15ms
    6:	learn: 0.2195026	total: 52.5ms	remaining: 7.5ms
    7:	learn: 0.1979265	total: 59.6ms	remaining: 0us
    Learning rate set to 0.218213
    0:	learn: 0.5824300	total: 9.53ms	remaining: 66.7ms
    1:	learn: 0.4304422	total: 16.5ms	remaining: 49.5ms
    2:	learn: 0.3418042	total: 23.4ms	remaining: 39ms
    3:	learn: 0.2766192	total: 30.1ms	remaining: 30.1ms
    4:	learn: 0.2448967	total: 36.7ms	remaining: 22ms
    5:	learn: 0.1725588	total: 43.3ms	remaining: 14.4ms
    6:	learn: 0.1251897	total: 53.3ms	remaining: 7.61ms
    7:	learn: 0.1093878	total: 60ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.5801132	total: 10.6ms	remaining: 84.9ms
    1:	learn: 0.4760646	total: 17.9ms	remaining: 62.6ms
    2:	learn: 0.3649597	total: 24.8ms	remaining: 49.6ms
    3:	learn: 0.2804805	total: 31.5ms	remaining: 39.4ms
    4:	learn: 0.2363755	total: 38.4ms	remaining: 30.7ms
    5:	learn: 0.1984137	total: 45.3ms	remaining: 22.7ms
    6:	learn: 0.1727929	total: 52.2ms	remaining: 14.9ms
    7:	learn: 0.1468732	total: 59ms	remaining: 7.38ms
    8:	learn: 0.1297278	total: 65.9ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.5981283	total: 6.69ms	remaining: 53.5ms
    1:	learn: 0.5264672	total: 13.3ms	remaining: 46.7ms
    2:	learn: 0.4541450	total: 19.9ms	remaining: 39.8ms
    3:	learn: 0.3963206	total: 26.5ms	remaining: 33.2ms
    4:	learn: 0.3473213	total: 33ms	remaining: 26.4ms
    5:	learn: 0.3224999	total: 39.7ms	remaining: 19.9ms
    6:	learn: 0.2604133	total: 46.6ms	remaining: 13.3ms
    7:	learn: 0.2230365	total: 53.5ms	remaining: 6.68ms
    8:	learn: 0.2097959	total: 60.4ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.5604963	total: 12.1ms	remaining: 96.6ms
    1:	learn: 0.4798653	total: 19.2ms	remaining: 67ms
    2:	learn: 0.3603238	total: 26ms	remaining: 52ms
    3:	learn: 0.2994041	total: 32.7ms	remaining: 40.9ms
    4:	learn: 0.2619298	total: 39.4ms	remaining: 31.5ms
    5:	learn: 0.2313410	total: 46ms	remaining: 23ms
    6:	learn: 0.2200122	total: 52.7ms	remaining: 15.1ms
    7:	learn: 0.1900323	total: 59.5ms	remaining: 7.44ms
    8:	learn: 0.1690382	total: 66.3ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.5558271	total: 6.83ms	remaining: 54.6ms
    1:	learn: 0.5064248	total: 13.7ms	remaining: 47.8ms
    2:	learn: 0.4310125	total: 20.3ms	remaining: 40.6ms
    3:	learn: 0.3669251	total: 27ms	remaining: 33.7ms
    4:	learn: 0.2820849	total: 33.7ms	remaining: 27ms
    5:	learn: 0.2496065	total: 40.5ms	remaining: 20.2ms
    6:	learn: 0.2206207	total: 47.8ms	remaining: 13.7ms
    7:	learn: 0.2018731	total: 54.7ms	remaining: 6.84ms
    8:	learn: 0.1700889	total: 62.4ms	remaining: 0us
    Learning rate set to 0.195872
    0:	learn: 0.5920972	total: 6.5ms	remaining: 52ms
    1:	learn: 0.4499923	total: 13.2ms	remaining: 46.1ms
    2:	learn: 0.3623142	total: 20ms	remaining: 39.9ms
    3:	learn: 0.2972499	total: 26.7ms	remaining: 33.4ms
    4:	learn: 0.2652394	total: 33.5ms	remaining: 26.8ms
    5:	learn: 0.1923362	total: 40ms	remaining: 20ms
    6:	learn: 0.1424651	total: 46.7ms	remaining: 13.3ms
    7:	learn: 0.1254946	total: 53.1ms	remaining: 6.64ms
    8:	learn: 0.1124844	total: 59.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5426157	total: 7ms	remaining: 21ms
    1:	learn: 0.4887303	total: 14.3ms	remaining: 14.3ms
    2:	learn: 0.4440783	total: 21.6ms	remaining: 7.19ms
    3:	learn: 0.4204197	total: 28.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4255143	total: 22.5ms	remaining: 67.4ms
    1:	learn: 0.3525834	total: 31.2ms	remaining: 31.2ms
    2:	learn: 0.2979189	total: 38.7ms	remaining: 12.9ms
    3:	learn: 0.2515309	total: 45.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5861366	total: 7.02ms	remaining: 21.1ms
    1:	learn: 0.4995383	total: 16.3ms	remaining: 16.3ms
    2:	learn: 0.4385082	total: 26.6ms	remaining: 8.86ms
    3:	learn: 0.3983389	total: 33.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5271283	total: 7.24ms	remaining: 21.7ms
    1:	learn: 0.4647182	total: 14.5ms	remaining: 14.5ms
    2:	learn: 0.4207615	total: 25.6ms	remaining: 8.54ms
    3:	learn: 0.3964177	total: 33.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4417777	total: 7.03ms	remaining: 21.1ms
    1:	learn: 0.3422761	total: 14.3ms	remaining: 14.3ms
    2:	learn: 0.3027255	total: 21.5ms	remaining: 7.18ms
    3:	learn: 0.2765251	total: 28.8ms	remaining: 0us
    Learning rate set to 0.457346
    0:	learn: 0.5262414	total: 19.6ms	remaining: 97.8ms
    1:	learn: 0.3855124	total: 46.7ms	remaining: 93.4ms
    2:	learn: 0.3378994	total: 61.7ms	remaining: 61.7ms
    3:	learn: 0.2901425	total: 76.4ms	remaining: 38.2ms
    4:	learn: 0.2535685	total: 91.5ms	remaining: 18.3ms
    5:	learn: 0.2056178	total: 106ms	remaining: 0us
    Learning rate set to 0.457346
    0:	learn: 0.4368990	total: 14.8ms	remaining: 73.8ms
    1:	learn: 0.3572616	total: 29.6ms	remaining: 59.2ms
    2:	learn: 0.3001696	total: 44.5ms	remaining: 44.5ms
    3:	learn: 0.2672472	total: 59.4ms	remaining: 29.7ms
    4:	learn: 0.2406167	total: 74.3ms	remaining: 14.9ms
    5:	learn: 0.1980230	total: 89.3ms	remaining: 0us
    Learning rate set to 0.457346
    0:	learn: 0.5516376	total: 14.5ms	remaining: 72.5ms
    1:	learn: 0.4379890	total: 29.5ms	remaining: 58.9ms
    2:	learn: 0.3836144	total: 44.3ms	remaining: 44.3ms
    3:	learn: 0.3072626	total: 59.2ms	remaining: 29.6ms
    4:	learn: 0.2565403	total: 74.5ms	remaining: 14.9ms
    5:	learn: 0.2428798	total: 91.6ms	remaining: 0us
    Learning rate set to 0.457346
    0:	learn: 0.5333846	total: 14.9ms	remaining: 74.3ms
    1:	learn: 0.4437349	total: 30ms	remaining: 60ms
    2:	learn: 0.4096080	total: 44.9ms	remaining: 44.9ms
    3:	learn: 0.3700645	total: 60.2ms	remaining: 30.1ms
    4:	learn: 0.3000062	total: 75.2ms	remaining: 15ms
    5:	learn: 0.2804895	total: 90.2ms	remaining: 0us
    Learning rate set to 0.457346
    0:	learn: 0.4264858	total: 14.5ms	remaining: 72.7ms
    1:	learn: 0.3253165	total: 31.1ms	remaining: 62.1ms
    2:	learn: 0.2801763	total: 45.6ms	remaining: 45.6ms
    3:	learn: 0.2414328	total: 60.3ms	remaining: 30.2ms
    4:	learn: 0.2036056	total: 74.8ms	remaining: 15ms
    5:	learn: 0.1681269	total: 89.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5426157	total: 7.03ms	remaining: 28.1ms
    1:	learn: 0.4887303	total: 14.3ms	remaining: 21.4ms
    2:	learn: 0.4440783	total: 21.4ms	remaining: 14.3ms
    3:	learn: 0.4204197	total: 28.7ms	remaining: 7.19ms
    4:	learn: 0.3720167	total: 36ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4255143	total: 8.03ms	remaining: 32.1ms
    1:	learn: 0.3525834	total: 15.1ms	remaining: 22.7ms
    2:	learn: 0.2979189	total: 22.7ms	remaining: 15.1ms
    3:	learn: 0.2515309	total: 30.2ms	remaining: 7.55ms
    4:	learn: 0.2455293	total: 37.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5861366	total: 7.1ms	remaining: 28.4ms
    1:	learn: 0.4995383	total: 14.8ms	remaining: 22.2ms
    2:	learn: 0.4385082	total: 22.2ms	remaining: 14.8ms
    3:	learn: 0.3983389	total: 29.6ms	remaining: 7.39ms
    4:	learn: 0.3742798	total: 36.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5271283	total: 6.98ms	remaining: 27.9ms
    1:	learn: 0.4647182	total: 14.2ms	remaining: 21.4ms
    2:	learn: 0.4207615	total: 21.5ms	remaining: 14.3ms
    3:	learn: 0.3964177	total: 28.7ms	remaining: 7.18ms
    4:	learn: 0.3469008	total: 35.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4417777	total: 11.9ms	remaining: 47.5ms
    1:	learn: 0.3422761	total: 19.9ms	remaining: 29.8ms
    2:	learn: 0.3027255	total: 27.1ms	remaining: 18.1ms
    3:	learn: 0.2765251	total: 34.4ms	remaining: 8.6ms
    4:	learn: 0.2461918	total: 41.7ms	remaining: 0us
    Learning rate set to 0.351298
    0:	learn: 0.5534801	total: 19.9ms	remaining: 139ms
    1:	learn: 0.4244460	total: 34.9ms	remaining: 105ms
    2:	learn: 0.3745693	total: 49.9ms	remaining: 83.1ms
    3:	learn: 0.3330919	total: 65.2ms	remaining: 65.2ms
    4:	learn: 0.2995758	total: 84.4ms	remaining: 50.6ms
    5:	learn: 0.2634999	total: 99.3ms	remaining: 33.1ms
    6:	learn: 0.2198488	total: 115ms	remaining: 16.5ms
    7:	learn: 0.2011351	total: 130ms	remaining: 0us
    Learning rate set to 0.351298
    0:	learn: 0.4738323	total: 14.5ms	remaining: 102ms
    1:	learn: 0.3941708	total: 29.7ms	remaining: 89.1ms
    2:	learn: 0.3442401	total: 44.7ms	remaining: 74.5ms
    3:	learn: 0.3082218	total: 59.6ms	remaining: 59.6ms
    4:	learn: 0.2656795	total: 74.8ms	remaining: 44.9ms
    5:	learn: 0.2370494	total: 89.9ms	remaining: 30ms
    6:	learn: 0.2194705	total: 105ms	remaining: 15ms
    7:	learn: 0.1993297	total: 120ms	remaining: 0us
    Learning rate set to 0.351298
    0:	learn: 0.5755581	total: 17.5ms	remaining: 122ms
    1:	learn: 0.4725976	total: 32.5ms	remaining: 97.4ms
    2:	learn: 0.4204664	total: 47.3ms	remaining: 78.8ms
    3:	learn: 0.3533080	total: 62.1ms	remaining: 62.1ms
    4:	learn: 0.3176940	total: 77.1ms	remaining: 46.3ms
    5:	learn: 0.2972167	total: 91.8ms	remaining: 30.6ms
    6:	learn: 0.2697353	total: 107ms	remaining: 15.2ms
    7:	learn: 0.2488905	total: 122ms	remaining: 0us
    Learning rate set to 0.351298
    0:	learn: 0.5590467	total: 14.6ms	remaining: 102ms
    1:	learn: 0.5138107	total: 29.8ms	remaining: 89.3ms
    2:	learn: 0.4637106	total: 44.5ms	remaining: 74.1ms
    3:	learn: 0.4172087	total: 63.6ms	remaining: 63.6ms
    4:	learn: 0.3698874	total: 78.5ms	remaining: 47.1ms
    5:	learn: 0.2999966	total: 93.3ms	remaining: 31.1ms
    6:	learn: 0.2842781	total: 108ms	remaining: 15.4ms
    7:	learn: 0.2629574	total: 123ms	remaining: 0us
    Learning rate set to 0.351298
    0:	learn: 0.4686362	total: 14.6ms	remaining: 102ms
    1:	learn: 0.3955000	total: 29.5ms	remaining: 88.5ms
    2:	learn: 0.3496346	total: 44.7ms	remaining: 74.6ms
    3:	learn: 0.3106725	total: 63.2ms	remaining: 63.2ms
    4:	learn: 0.2744132	total: 78ms	remaining: 46.8ms
    5:	learn: 0.2295465	total: 94.5ms	remaining: 31.5ms
    6:	learn: 0.2192365	total: 110ms	remaining: 15.6ms
    7:	learn: 0.2022441	total: 127ms	remaining: 0us
    Learning rate set to 0.315333
    0:	learn: 0.5641671	total: 14.8ms	remaining: 118ms
    1:	learn: 0.4407212	total: 38.5ms	remaining: 135ms
    2:	learn: 0.3902084	total: 59.5ms	remaining: 119ms
    3:	learn: 0.3489555	total: 74.4ms	remaining: 93ms
    4:	learn: 0.2762414	total: 89.5ms	remaining: 71.6ms
    5:	learn: 0.2526831	total: 104ms	remaining: 52.1ms
    6:	learn: 0.2112256	total: 119ms	remaining: 34ms
    7:	learn: 0.2054572	total: 134ms	remaining: 16.7ms
    8:	learn: 0.1919609	total: 148ms	remaining: 0us
    Learning rate set to 0.315333
    0:	learn: 0.4890235	total: 18.8ms	remaining: 150ms
    1:	learn: 0.4088036	total: 34.1ms	remaining: 119ms
    2:	learn: 0.3791440	total: 49.2ms	remaining: 98.5ms
    3:	learn: 0.3386908	total: 64.3ms	remaining: 80.4ms
    4:	learn: 0.2935622	total: 80.2ms	remaining: 64.1ms
    5:	learn: 0.2558632	total: 96.1ms	remaining: 48.1ms
    6:	learn: 0.2300302	total: 111ms	remaining: 31.7ms
    7:	learn: 0.2049432	total: 126ms	remaining: 15.8ms
    8:	learn: 0.1780776	total: 141ms	remaining: 0us
    Learning rate set to 0.315333
    0:	learn: 0.5848337	total: 14.8ms	remaining: 118ms
    1:	learn: 0.4868915	total: 29.9ms	remaining: 105ms
    2:	learn: 0.4358312	total: 45ms	remaining: 90.1ms
    3:	learn: 0.3719898	total: 59.9ms	remaining: 74.9ms
    4:	learn: 0.3369241	total: 74.8ms	remaining: 59.9ms
    5:	learn: 0.3162342	total: 89.7ms	remaining: 44.8ms
    6:	learn: 0.2894754	total: 105ms	remaining: 29.9ms
    7:	learn: 0.2690282	total: 119ms	remaining: 14.9ms
    8:	learn: 0.2465817	total: 134ms	remaining: 0us
    Learning rate set to 0.315333
    0:	learn: 0.5691932	total: 14.7ms	remaining: 118ms
    1:	learn: 0.5250655	total: 29.5ms	remaining: 103ms
    2:	learn: 0.4760143	total: 44.6ms	remaining: 89.2ms
    3:	learn: 0.4306848	total: 59.3ms	remaining: 74.1ms
    4:	learn: 0.3855873	total: 74ms	remaining: 59.2ms
    5:	learn: 0.3171644	total: 88.8ms	remaining: 44.4ms
    6:	learn: 0.3018473	total: 104ms	remaining: 29.6ms
    7:	learn: 0.2815111	total: 118ms	remaining: 14.8ms
    8:	learn: 0.2709289	total: 133ms	remaining: 0us
    Learning rate set to 0.315333
    0:	learn: 0.4854397	total: 14.7ms	remaining: 117ms
    1:	learn: 0.4122464	total: 29.6ms	remaining: 104ms
    2:	learn: 0.3664093	total: 44.4ms	remaining: 88.8ms
    3:	learn: 0.3285411	total: 59.4ms	remaining: 74.3ms
    4:	learn: 0.2768637	total: 75ms	remaining: 60ms
    5:	learn: 0.2329384	total: 90.2ms	remaining: 45.1ms
    6:	learn: 0.2244032	total: 105ms	remaining: 30ms
    7:	learn: 0.2084638	total: 120ms	remaining: 15ms
    8:	learn: 0.1796759	total: 135ms	remaining: 0us
    Learning rate set to 0.397059
    0:	learn: 0.5409593	total: 14.8ms	remaining: 88.7ms
    1:	learn: 0.4061299	total: 29.6ms	remaining: 74.1ms
    2:	learn: 0.3572280	total: 44.7ms	remaining: 59.6ms
    3:	learn: 0.3131450	total: 59.6ms	remaining: 44.7ms
    4:	learn: 0.2786438	total: 75ms	remaining: 30ms
    5:	learn: 0.2431045	total: 90ms	remaining: 15ms
    6:	learn: 0.1999922	total: 105ms	remaining: 0us
    Learning rate set to 0.397059
    0:	learn: 0.4565216	total: 14.8ms	remaining: 88.7ms
    1:	learn: 0.3784574	total: 30.7ms	remaining: 76.7ms
    2:	learn: 0.3272554	total: 55.5ms	remaining: 74ms
    3:	learn: 0.2901650	total: 70.7ms	remaining: 53ms
    4:	learn: 0.2458654	total: 86.2ms	remaining: 34.5ms
    5:	learn: 0.2158585	total: 101ms	remaining: 16.9ms
    6:	learn: 0.1981233	total: 117ms	remaining: 0us
    Learning rate set to 0.397059
    0:	learn: 0.5646133	total: 14.4ms	remaining: 86.6ms
    1:	learn: 0.4563801	total: 29.3ms	remaining: 73.1ms
    2:	learn: 0.4031776	total: 44.1ms	remaining: 58.9ms
    3:	learn: 0.3319511	total: 60.1ms	remaining: 45.1ms
    4:	learn: 0.2825773	total: 75ms	remaining: 30ms
    5:	learn: 0.2597739	total: 90ms	remaining: 15ms
    6:	learn: 0.2183364	total: 108ms	remaining: 0us
    Learning rate set to 0.397059
    0:	learn: 0.5472083	total: 17ms	remaining: 102ms
    1:	learn: 0.5010595	total: 32.1ms	remaining: 80.2ms
    2:	learn: 0.4428718	total: 47.3ms	remaining: 63.1ms
    3:	learn: 0.3944631	total: 62.6ms	remaining: 46.9ms
    4:	learn: 0.3200708	total: 77.6ms	remaining: 31ms
    5:	learn: 0.2867900	total: 92.8ms	remaining: 15.5ms
    6:	learn: 0.2666057	total: 108ms	remaining: 0us
    Learning rate set to 0.397059
    0:	learn: 0.4491352	total: 14.7ms	remaining: 88.1ms
    1:	learn: 0.3770873	total: 29.7ms	remaining: 74.3ms
    2:	learn: 0.3297325	total: 44.5ms	remaining: 59.3ms
    3:	learn: 0.2844499	total: 59.2ms	remaining: 44.4ms
    4:	learn: 0.2410640	total: 73.9ms	remaining: 29.6ms
    5:	learn: 0.2073199	total: 90.3ms	remaining: 15.1ms
    6:	learn: 0.1904341	total: 105ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5305433	total: 29.5ms	remaining: 207ms
    1:	learn: 0.4455605	total: 59.4ms	remaining: 178ms
    2:	learn: 0.3555325	total: 89.3ms	remaining: 149ms
    3:	learn: 0.3097568	total: 121ms	remaining: 121ms
    4:	learn: 0.2774929	total: 153ms	remaining: 91.6ms
    5:	learn: 0.2505432	total: 184ms	remaining: 61.2ms
    6:	learn: 0.2230931	total: 219ms	remaining: 31.3ms
    7:	learn: 0.2003947	total: 249ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5202548	total: 36.7ms	remaining: 257ms
    1:	learn: 0.4516423	total: 67.9ms	remaining: 204ms
    2:	learn: 0.3925906	total: 97.8ms	remaining: 163ms
    3:	learn: 0.3409991	total: 128ms	remaining: 128ms
    4:	learn: 0.3000774	total: 158ms	remaining: 94.8ms
    5:	learn: 0.2785641	total: 204ms	remaining: 68ms
    6:	learn: 0.2423316	total: 235ms	remaining: 33.6ms
    7:	learn: 0.2256805	total: 266ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5102174	total: 29.2ms	remaining: 205ms
    1:	learn: 0.4166163	total: 58.8ms	remaining: 176ms
    2:	learn: 0.3748692	total: 88.8ms	remaining: 148ms
    3:	learn: 0.3336462	total: 119ms	remaining: 119ms
    4:	learn: 0.2751407	total: 154ms	remaining: 92.3ms
    5:	learn: 0.2572054	total: 185ms	remaining: 61.8ms
    6:	learn: 0.2240356	total: 223ms	remaining: 31.8ms
    7:	learn: 0.2091368	total: 253ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5542113	total: 31.6ms	remaining: 221ms
    1:	learn: 0.4522021	total: 61.3ms	remaining: 184ms
    2:	learn: 0.3815010	total: 93.3ms	remaining: 155ms
    3:	learn: 0.3281179	total: 123ms	remaining: 123ms
    4:	learn: 0.2867375	total: 154ms	remaining: 92.2ms
    5:	learn: 0.2604733	total: 184ms	remaining: 61.2ms
    6:	learn: 0.2378808	total: 223ms	remaining: 31.8ms
    7:	learn: 0.2228247	total: 258ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5479062	total: 29.6ms	remaining: 207ms
    1:	learn: 0.4296088	total: 58.6ms	remaining: 176ms
    2:	learn: 0.3888738	total: 87.7ms	remaining: 146ms
    3:	learn: 0.3501726	total: 117ms	remaining: 117ms
    4:	learn: 0.2825685	total: 147ms	remaining: 88.2ms
    5:	learn: 0.2503976	total: 178ms	remaining: 59.2ms
    6:	learn: 0.2204463	total: 212ms	remaining: 30.3ms
    7:	learn: 0.2080867	total: 244ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5305433	total: 29.4ms	remaining: 235ms
    1:	learn: 0.4455605	total: 62.4ms	remaining: 218ms
    2:	learn: 0.3555325	total: 91.7ms	remaining: 183ms
    3:	learn: 0.3097568	total: 124ms	remaining: 155ms
    4:	learn: 0.2774929	total: 153ms	remaining: 123ms
    5:	learn: 0.2505432	total: 183ms	remaining: 91.4ms
    6:	learn: 0.2230931	total: 221ms	remaining: 63.2ms
    7:	learn: 0.2003947	total: 251ms	remaining: 31.4ms
    8:	learn: 0.1811005	total: 293ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5202548	total: 29.5ms	remaining: 236ms
    1:	learn: 0.4516423	total: 59.8ms	remaining: 209ms
    2:	learn: 0.3925906	total: 90.1ms	remaining: 180ms
    3:	learn: 0.3409991	total: 123ms	remaining: 154ms
    4:	learn: 0.3000774	total: 157ms	remaining: 126ms
    5:	learn: 0.2785641	total: 189ms	remaining: 94.3ms
    6:	learn: 0.2423316	total: 225ms	remaining: 64.3ms
    7:	learn: 0.2256805	total: 255ms	remaining: 31.9ms
    8:	learn: 0.2136524	total: 285ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5102174	total: 29ms	remaining: 232ms
    1:	learn: 0.4166163	total: 60.6ms	remaining: 212ms
    2:	learn: 0.3748692	total: 90.1ms	remaining: 180ms
    3:	learn: 0.3336462	total: 120ms	remaining: 150ms
    4:	learn: 0.2751407	total: 150ms	remaining: 120ms
    5:	learn: 0.2572054	total: 179ms	remaining: 89.6ms
    6:	learn: 0.2240356	total: 214ms	remaining: 61.2ms
    7:	learn: 0.2091368	total: 254ms	remaining: 31.8ms
    8:	learn: 0.1733750	total: 285ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5542113	total: 29ms	remaining: 232ms
    1:	learn: 0.4522021	total: 59ms	remaining: 206ms
    2:	learn: 0.3815010	total: 88.5ms	remaining: 177ms
    3:	learn: 0.3281179	total: 118ms	remaining: 148ms
    4:	learn: 0.2867375	total: 151ms	remaining: 120ms
    5:	learn: 0.2604733	total: 182ms	remaining: 90.9ms
    6:	learn: 0.2378808	total: 226ms	remaining: 64.5ms
    7:	learn: 0.2228247	total: 257ms	remaining: 32.2ms
    8:	learn: 0.2014161	total: 287ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5479062	total: 29.5ms	remaining: 236ms
    1:	learn: 0.4296088	total: 70.3ms	remaining: 246ms
    2:	learn: 0.3888738	total: 99.6ms	remaining: 199ms
    3:	learn: 0.3501726	total: 129ms	remaining: 162ms
    4:	learn: 0.2825685	total: 159ms	remaining: 127ms
    5:	learn: 0.2503976	total: 189ms	remaining: 94.5ms
    6:	learn: 0.2204463	total: 226ms	remaining: 64.5ms
    7:	learn: 0.2080867	total: 259ms	remaining: 32.4ms
    8:	learn: 0.1827759	total: 289ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5555983	total: 32.3ms	remaining: 259ms
    1:	learn: 0.4565304	total: 63.3ms	remaining: 221ms
    2:	learn: 0.3843872	total: 94.6ms	remaining: 189ms
    3:	learn: 0.3368834	total: 124ms	remaining: 155ms
    4:	learn: 0.2860435	total: 154ms	remaining: 123ms
    5:	learn: 0.2659767	total: 184ms	remaining: 91.8ms
    6:	learn: 0.2355804	total: 219ms	remaining: 62.6ms
    7:	learn: 0.2260181	total: 250ms	remaining: 31.2ms
    8:	learn: 0.1898352	total: 279ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5958881	total: 6.33ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6373882	total: 3.45ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6723374	total: 3.57ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5656372	total: 3.46ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6110258	total: 3.38ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5958881	total: 3.59ms	remaining: 3.59ms
    1:	learn: 0.4538711	total: 7.43ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6373882	total: 3.48ms	remaining: 3.48ms
    1:	learn: 0.5116494	total: 6.76ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6723374	total: 3.64ms	remaining: 3.64ms
    1:	learn: 0.5135140	total: 7.52ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5656372	total: 3.49ms	remaining: 3.49ms
    1:	learn: 0.4800094	total: 7.34ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6110258	total: 3.6ms	remaining: 3.6ms
    1:	learn: 0.4764225	total: 7.34ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5958881	total: 3.59ms	remaining: 7.17ms
    1:	learn: 0.4538711	total: 7.47ms	remaining: 3.74ms
    2:	learn: 0.4192034	total: 11.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6373882	total: 3.45ms	remaining: 6.89ms
    1:	learn: 0.5116494	total: 7.31ms	remaining: 3.66ms
    2:	learn: 0.3958742	total: 10.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6723374	total: 3.51ms	remaining: 7.02ms
    1:	learn: 0.5135140	total: 7.29ms	remaining: 3.64ms
    2:	learn: 0.4131307	total: 11.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5656372	total: 3.46ms	remaining: 6.92ms
    1:	learn: 0.4800094	total: 7.28ms	remaining: 3.64ms
    2:	learn: 0.4044282	total: 11.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.6110258	total: 3.57ms	remaining: 7.15ms
    1:	learn: 0.4764225	total: 7.44ms	remaining: 3.72ms
    2:	learn: 0.3412531	total: 11.4ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.6050986	total: 3.61ms	remaining: 10.8ms
    1:	learn: 0.4685951	total: 7.52ms	remaining: 7.52ms
    2:	learn: 0.4328948	total: 11.3ms	remaining: 3.77ms
    3:	learn: 0.3622974	total: 15.1ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.6428680	total: 3.49ms	remaining: 10.5ms
    1:	learn: 0.5231921	total: 7.29ms	remaining: 7.29ms
    2:	learn: 0.4165256	total: 11ms	remaining: 3.65ms
    3:	learn: 0.3319462	total: 14.5ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.6744290	total: 4.71ms	remaining: 14.1ms
    1:	learn: 0.5290140	total: 11.2ms	remaining: 11.2ms
    2:	learn: 0.4508841	total: 15ms	remaining: 5ms
    3:	learn: 0.4046275	total: 18.7ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.5777614	total: 3.52ms	remaining: 10.5ms
    1:	learn: 0.4947627	total: 7.28ms	remaining: 7.28ms
    2:	learn: 0.4238172	total: 11ms	remaining: 3.67ms
    3:	learn: 0.3641860	total: 14.7ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.6190464	total: 3.46ms	remaining: 10.4ms
    1:	learn: 0.4880668	total: 7.36ms	remaining: 7.36ms
    2:	learn: 0.3620279	total: 11.1ms	remaining: 3.7ms
    3:	learn: 0.2457087	total: 14.8ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.6171278	total: 3.6ms	remaining: 14.4ms
    1:	learn: 0.4901220	total: 7.48ms	remaining: 11.2ms
    2:	learn: 0.4533037	total: 11.3ms	remaining: 7.54ms
    3:	learn: 0.3629723	total: 15.1ms	remaining: 3.77ms
    4:	learn: 0.3497858	total: 20ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.6499412	total: 3.44ms	remaining: 13.8ms
    1:	learn: 0.5399947	total: 7.15ms	remaining: 10.7ms
    2:	learn: 0.4599687	total: 10.8ms	remaining: 7.17ms
    3:	learn: 0.3638647	total: 14.3ms	remaining: 3.58ms
    4:	learn: 0.3303280	total: 17.9ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.6771028	total: 3.79ms	remaining: 15.2ms
    1:	learn: 0.5496813	total: 7.67ms	remaining: 11.5ms
    2:	learn: 0.4757461	total: 11.4ms	remaining: 7.63ms
    3:	learn: 0.4341504	total: 15.1ms	remaining: 3.79ms
    4:	learn: 0.3900098	total: 18.7ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.5935920	total: 3.77ms	remaining: 15.1ms
    1:	learn: 0.5160376	total: 8.85ms	remaining: 13.3ms
    2:	learn: 0.4643756	total: 12.7ms	remaining: 8.45ms
    3:	learn: 0.4187818	total: 16.5ms	remaining: 4.12ms
    4:	learn: 0.3552957	total: 20.3ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.6294250	total: 3.77ms	remaining: 15.1ms
    1:	learn: 0.5056919	total: 7.64ms	remaining: 11.5ms
    2:	learn: 0.3906547	total: 11.3ms	remaining: 7.54ms
    3:	learn: 0.2797623	total: 14.9ms	remaining: 3.73ms
    4:	learn: 0.2620094	total: 18.5ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.6263482	total: 3.68ms	remaining: 18.4ms
    1:	learn: 0.5084854	total: 7.72ms	remaining: 15.4ms
    2:	learn: 0.4712208	total: 11.5ms	remaining: 11.5ms
    3:	learn: 0.4141641	total: 15.3ms	remaining: 7.65ms
    4:	learn: 0.3933750	total: 19.3ms	remaining: 3.85ms
    5:	learn: 0.3599194	total: 22.9ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.6553055	total: 3.45ms	remaining: 17.2ms
    1:	learn: 0.5542407	total: 7.24ms	remaining: 14.5ms
    2:	learn: 0.4784473	total: 11ms	remaining: 11ms
    3:	learn: 0.3909149	total: 14.7ms	remaining: 7.36ms
    4:	learn: 0.3583560	total: 18.4ms	remaining: 3.68ms
    5:	learn: 0.3488278	total: 22.2ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.6791240	total: 3.65ms	remaining: 18.2ms
    1:	learn: 0.6392969	total: 7.55ms	remaining: 15.1ms
    2:	learn: 0.5927846	total: 11.3ms	remaining: 11.3ms
    3:	learn: 0.5584257	total: 15ms	remaining: 7.52ms
    4:	learn: 0.5083209	total: 18.8ms	remaining: 3.76ms
    5:	learn: 0.4642083	total: 22.6ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.6057105	total: 3.42ms	remaining: 17.1ms
    1:	learn: 0.5336815	total: 7.1ms	remaining: 14.2ms
    2:	learn: 0.4865698	total: 10.7ms	remaining: 10.7ms
    3:	learn: 0.4434462	total: 14.4ms	remaining: 7.18ms
    4:	learn: 0.3832190	total: 18.1ms	remaining: 3.61ms
    5:	learn: 0.3404078	total: 21.9ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.6373144	total: 3.47ms	remaining: 17.3ms
    1:	learn: 0.5212571	total: 7.2ms	remaining: 14.4ms
    2:	learn: 0.4144480	total: 10.8ms	remaining: 10.8ms
    3:	learn: 0.3084565	total: 14.5ms	remaining: 7.24ms
    4:	learn: 0.2904632	total: 18.1ms	remaining: 3.62ms
    5:	learn: 0.2777373	total: 21.7ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.6335756	total: 3.92ms	remaining: 23.5ms
    1:	learn: 0.5240554	total: 7.88ms	remaining: 19.7ms
    2:	learn: 0.4868755	total: 11.8ms	remaining: 15.7ms
    3:	learn: 0.4335603	total: 17.2ms	remaining: 12.9ms
    4:	learn: 0.4125754	total: 25ms	remaining: 10ms
    5:	learn: 0.3801868	total: 28.8ms	remaining: 4.79ms
    6:	learn: 0.3193604	total: 32.4ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.6594784	total: 3.51ms	remaining: 21.1ms
    1:	learn: 0.5662514	total: 7.32ms	remaining: 18.3ms
    2:	learn: 0.4942516	total: 11ms	remaining: 14.7ms
    3:	learn: 0.4133881	total: 14.6ms	remaining: 11ms
    4:	learn: 0.3818431	total: 18.3ms	remaining: 7.31ms
    5:	learn: 0.3738515	total: 24ms	remaining: 4ms
    6:	learn: 0.3318603	total: 27.7ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.6806899	total: 3.52ms	remaining: 21.1ms
    1:	learn: 0.6444611	total: 7.33ms	remaining: 18.3ms
    2:	learn: 0.6026570	total: 11.1ms	remaining: 14.8ms
    3:	learn: 0.5459564	total: 15.2ms	remaining: 11.4ms
    4:	learn: 0.4763820	total: 19ms	remaining: 7.58ms
    5:	learn: 0.3925903	total: 22.7ms	remaining: 3.78ms
    6:	learn: 0.3415358	total: 26.4ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.6152073	total: 3.48ms	remaining: 20.9ms
    1:	learn: 0.5482993	total: 7.98ms	remaining: 20ms
    2:	learn: 0.5049111	total: 13.5ms	remaining: 18ms
    3:	learn: 0.4640228	total: 17.1ms	remaining: 12.9ms
    4:	learn: 0.4068421	total: 20.9ms	remaining: 8.36ms
    5:	learn: 0.3671253	total: 24.6ms	remaining: 4.1ms
    6:	learn: 0.3192009	total: 29.6ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.6434588	total: 3.54ms	remaining: 21.2ms
    1:	learn: 0.5347925	total: 7.3ms	remaining: 18.2ms
    2:	learn: 0.4346585	total: 11.1ms	remaining: 14.7ms
    3:	learn: 0.3330804	total: 14.6ms	remaining: 11ms
    4:	learn: 0.2848408	total: 18.2ms	remaining: 7.29ms
    5:	learn: 0.2737807	total: 21.6ms	remaining: 3.6ms
    6:	learn: 0.2364995	total: 25.2ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.6393724	total: 3.65ms	remaining: 25.5ms
    1:	learn: 0.5373031	total: 7.62ms	remaining: 22.9ms
    2:	learn: 0.5005843	total: 11.6ms	remaining: 19.3ms
    3:	learn: 0.4501635	total: 15.6ms	remaining: 15.6ms
    4:	learn: 0.4290270	total: 19.5ms	remaining: 11.7ms
    5:	learn: 0.3975166	total: 24.5ms	remaining: 8.15ms
    6:	learn: 0.3377418	total: 28.4ms	remaining: 4.05ms
    7:	learn: 0.3260663	total: 32.3ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.6628144	total: 3.56ms	remaining: 24.9ms
    1:	learn: 0.5764239	total: 7.39ms	remaining: 22.2ms
    2:	learn: 0.5079127	total: 13.7ms	remaining: 22.8ms
    3:	learn: 0.4324430	total: 19.5ms	remaining: 19.5ms
    4:	learn: 0.4018355	total: 23.1ms	remaining: 13.9ms
    5:	learn: 0.3948125	total: 26.8ms	remaining: 8.94ms
    6:	learn: 0.3535157	total: 30.4ms	remaining: 4.34ms
    7:	learn: 0.3191633	total: 33.9ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.6819341	total: 3.54ms	remaining: 24.8ms
    1:	learn: 0.6486660	total: 7.3ms	remaining: 21.9ms
    2:	learn: 0.5744201	total: 11ms	remaining: 18.3ms
    3:	learn: 0.5388337	total: 14.7ms	remaining: 14.7ms
    4:	learn: 0.4863156	total: 18.2ms	remaining: 10.9ms
    5:	learn: 0.4456397	total: 21.8ms	remaining: 7.28ms
    6:	learn: 0.4249167	total: 25.6ms	remaining: 3.65ms
    7:	learn: 0.3968416	total: 29.3ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.6228162	total: 5.52ms	remaining: 38.6ms
    1:	learn: 0.5605034	total: 10.9ms	remaining: 32.6ms
    2:	learn: 0.5202427	total: 14.8ms	remaining: 24.6ms
    3:	learn: 0.4470358	total: 18.6ms	remaining: 18.6ms
    4:	learn: 0.3966679	total: 25.2ms	remaining: 15.1ms
    5:	learn: 0.3734550	total: 32.6ms	remaining: 10.9ms
    6:	learn: 0.3259843	total: 36.3ms	remaining: 5.18ms
    7:	learn: 0.3099205	total: 39.8ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.6483701	total: 3.43ms	remaining: 24ms
    1:	learn: 0.5465275	total: 7.19ms	remaining: 21.6ms
    2:	learn: 0.4520861	total: 10.8ms	remaining: 17.9ms
    3:	learn: 0.3545347	total: 14.2ms	remaining: 14.2ms
    4:	learn: 0.3084692	total: 17.8ms	remaining: 10.7ms
    5:	learn: 0.2972044	total: 21.3ms	remaining: 7.09ms
    6:	learn: 0.2570870	total: 24.8ms	remaining: 3.55ms
    7:	learn: 0.2165252	total: 28.2ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.6441218	total: 3.56ms	remaining: 28.5ms
    1:	learn: 0.5486701	total: 7.41ms	remaining: 25.9ms
    2:	learn: 0.5126443	total: 11.1ms	remaining: 22.2ms
    3:	learn: 0.4502481	total: 14.7ms	remaining: 18.4ms
    4:	learn: 0.4296666	total: 18.5ms	remaining: 14.8ms
    5:	learn: 0.4024748	total: 22.2ms	remaining: 11.1ms
    6:	learn: 0.3508216	total: 25.8ms	remaining: 7.37ms
    7:	learn: 0.3420810	total: 29.5ms	remaining: 3.69ms
    8:	learn: 0.3280021	total: 33.2ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.6655358	total: 3.57ms	remaining: 28.6ms
    1:	learn: 0.5851114	total: 7.2ms	remaining: 25.2ms
    2:	learn: 0.5198335	total: 11.6ms	remaining: 23.2ms
    3:	learn: 0.4488645	total: 21.3ms	remaining: 26.6ms
    4:	learn: 0.4191200	total: 28.3ms	remaining: 22.6ms
    5:	learn: 0.4127016	total: 32.7ms	remaining: 16.4ms
    6:	learn: 0.3722434	total: 36.6ms	remaining: 10.4ms
    7:	learn: 0.3400277	total: 40.2ms	remaining: 5.03ms
    8:	learn: 0.3253185	total: 43.7ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.6829473	total: 3.55ms	remaining: 28.4ms
    1:	learn: 0.6522051	total: 7.38ms	remaining: 25.8ms
    2:	learn: 0.5837797	total: 11.1ms	remaining: 22.2ms
    3:	learn: 0.5507024	total: 14.8ms	remaining: 18.5ms
    4:	learn: 0.5015654	total: 18.4ms	remaining: 14.7ms
    5:	learn: 0.4620186	total: 22.2ms	remaining: 11.1ms
    6:	learn: 0.4282262	total: 25.9ms	remaining: 7.39ms
    7:	learn: 0.4072883	total: 29.4ms	remaining: 3.68ms
    8:	learn: 0.3945914	total: 33.1ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.6290474	total: 3.43ms	remaining: 27.5ms
    1:	learn: 0.5707975	total: 7.16ms	remaining: 25.1ms
    2:	learn: 0.5332256	total: 10.8ms	remaining: 21.7ms
    3:	learn: 0.4640259	total: 14.5ms	remaining: 18.1ms
    4:	learn: 0.4156609	total: 18.2ms	remaining: 14.5ms
    5:	learn: 0.3930780	total: 21.8ms	remaining: 10.9ms
    6:	learn: 0.3469386	total: 26.1ms	remaining: 7.45ms
    7:	learn: 0.3319669	total: 30.1ms	remaining: 3.76ms
    8:	learn: 0.3114600	total: 33.7ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.6523778	total: 3.44ms	remaining: 27.5ms
    1:	learn: 0.5567239	total: 7.21ms	remaining: 25.2ms
    2:	learn: 0.4672916	total: 18.2ms	remaining: 36.3ms
    3:	learn: 0.3734641	total: 23.8ms	remaining: 29.8ms
    4:	learn: 0.3603351	total: 32.8ms	remaining: 26.3ms
    5:	learn: 0.3480546	total: 37.6ms	remaining: 18.8ms
    6:	learn: 0.3005921	total: 41ms	remaining: 11.7ms
    7:	learn: 0.2556018	total: 44.6ms	remaining: 5.57ms
    8:	learn: 0.2336415	total: 49.1ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4706299	total: 7.49ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5736882	total: 6.84ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5455128	total: 7.13ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4939307	total: 6.87ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4817373	total: 6.93ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4706299	total: 7.39ms	remaining: 7.39ms
    1:	learn: 0.3719182	total: 16ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5736882	total: 6.86ms	remaining: 6.86ms
    1:	learn: 0.3839978	total: 13.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5455128	total: 7.14ms	remaining: 7.14ms
    1:	learn: 0.3433948	total: 16.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4939307	total: 7.06ms	remaining: 7.06ms
    1:	learn: 0.2653886	total: 14.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4817373	total: 6.76ms	remaining: 6.76ms
    1:	learn: 0.2827232	total: 13.6ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4706299	total: 7.21ms	remaining: 14.4ms
    1:	learn: 0.3719182	total: 14.7ms	remaining: 7.36ms
    2:	learn: 0.3057157	total: 22.2ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5736882	total: 7.06ms	remaining: 14.1ms
    1:	learn: 0.3839978	total: 14.5ms	remaining: 7.25ms
    2:	learn: 0.2782609	total: 21.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5455128	total: 7.09ms	remaining: 14.2ms
    1:	learn: 0.3433948	total: 14.6ms	remaining: 7.29ms
    2:	learn: 0.2371878	total: 21.9ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4939307	total: 6.75ms	remaining: 13.5ms
    1:	learn: 0.2653886	total: 13.8ms	remaining: 6.88ms
    2:	learn: 0.1928014	total: 20.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4817373	total: 7.21ms	remaining: 14.4ms
    1:	learn: 0.2827232	total: 13.8ms	remaining: 6.89ms
    2:	learn: 0.1748453	total: 20.7ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.4908129	total: 7.22ms	remaining: 21.7ms
    1:	learn: 0.3980294	total: 14.7ms	remaining: 14.7ms
    2:	learn: 0.3350375	total: 22.3ms	remaining: 7.43ms
    3:	learn: 0.2373178	total: 29.8ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.5849086	total: 6.99ms	remaining: 21ms
    1:	learn: 0.4088328	total: 14ms	remaining: 14ms
    2:	learn: 0.3209856	total: 21ms	remaining: 6.99ms
    3:	learn: 0.2532025	total: 33.2ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.5575730	total: 7.04ms	remaining: 21.1ms
    1:	learn: 0.3708511	total: 14.1ms	remaining: 14.1ms
    2:	learn: 0.2661002	total: 21.1ms	remaining: 7.02ms
    3:	learn: 0.2315772	total: 28.5ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.5129070	total: 6.95ms	remaining: 20.8ms
    1:	learn: 0.2948875	total: 14.1ms	remaining: 14.1ms
    2:	learn: 0.2200154	total: 21.1ms	remaining: 7.02ms
    3:	learn: 0.1705502	total: 28ms	remaining: 0us
    Learning rate set to 0.42914
    0:	learn: 0.5001173	total: 6.88ms	remaining: 20.6ms
    1:	learn: 0.3074153	total: 14.1ms	remaining: 14.1ms
    2:	learn: 0.2079372	total: 23.4ms	remaining: 7.8ms
    3:	learn: 0.1600504	total: 34.7ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.5174762	total: 7.17ms	remaining: 28.7ms
    1:	learn: 0.4422750	total: 14.4ms	remaining: 21.6ms
    2:	learn: 0.3756775	total: 21.6ms	remaining: 14.4ms
    3:	learn: 0.3185479	total: 28.8ms	remaining: 7.19ms
    4:	learn: 0.2719303	total: 36.2ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.5995576	total: 9.36ms	remaining: 37.5ms
    1:	learn: 0.4419019	total: 18.7ms	remaining: 28.1ms
    2:	learn: 0.3577740	total: 26.5ms	remaining: 17.7ms
    3:	learn: 0.2927705	total: 33.1ms	remaining: 8.27ms
    4:	learn: 0.2411816	total: 39.7ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.5739820	total: 8.33ms	remaining: 33.3ms
    1:	learn: 0.4076298	total: 25.3ms	remaining: 38ms
    2:	learn: 0.3064263	total: 33.3ms	remaining: 22.2ms
    3:	learn: 0.2748521	total: 40.6ms	remaining: 10.1ms
    4:	learn: 0.2187821	total: 47.9ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.5375982	total: 6.89ms	remaining: 27.5ms
    1:	learn: 0.3363142	total: 14ms	remaining: 21ms
    2:	learn: 0.2666617	total: 20.8ms	remaining: 13.9ms
    3:	learn: 0.2135814	total: 27.8ms	remaining: 6.95ms
    4:	learn: 0.1762280	total: 41.5ms	remaining: 0us
    Learning rate set to 0.34973
    0:	learn: 0.5248642	total: 7.17ms	remaining: 28.7ms
    1:	learn: 0.3434296	total: 14.4ms	remaining: 21.6ms
    2:	learn: 0.2057583	total: 21.6ms	remaining: 14.4ms
    3:	learn: 0.1493749	total: 28.8ms	remaining: 7.19ms
    4:	learn: 0.1096182	total: 35.9ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.5381315	total: 7.12ms	remaining: 35.6ms
    1:	learn: 0.4662480	total: 14.4ms	remaining: 28.9ms
    2:	learn: 0.3980208	total: 21.6ms	remaining: 21.6ms
    3:	learn: 0.3272761	total: 28.8ms	remaining: 14.4ms
    4:	learn: 0.2862064	total: 35.8ms	remaining: 7.16ms
    5:	learn: 0.2310235	total: 42.9ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.6107980	total: 6.92ms	remaining: 34.6ms
    1:	learn: 0.4681246	total: 19.5ms	remaining: 38.9ms
    2:	learn: 0.3878570	total: 31.5ms	remaining: 31.5ms
    3:	learn: 0.3252131	total: 38.6ms	remaining: 19.3ms
    4:	learn: 0.2847424	total: 46ms	remaining: 9.2ms
    5:	learn: 0.2354983	total: 55ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.5870706	total: 7ms	remaining: 35ms
    1:	learn: 0.4369168	total: 13.7ms	remaining: 27.5ms
    2:	learn: 0.3398789	total: 20.5ms	remaining: 20.5ms
    3:	learn: 0.3010241	total: 27.3ms	remaining: 13.6ms
    4:	learn: 0.2477192	total: 34.3ms	remaining: 6.86ms
    5:	learn: 0.2227051	total: 41.3ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.5564722	total: 6.79ms	remaining: 34ms
    1:	learn: 0.3705035	total: 18.2ms	remaining: 36.4ms
    2:	learn: 0.3054388	total: 28.5ms	remaining: 28.5ms
    3:	learn: 0.2539115	total: 35.5ms	remaining: 17.7ms
    4:	learn: 0.2076793	total: 42.5ms	remaining: 8.51ms
    5:	learn: 0.1733768	total: 49.3ms	remaining: 0us
    Learning rate set to 0.295885
    0:	learn: 0.5442852	total: 6.96ms	remaining: 34.8ms
    1:	learn: 0.3742029	total: 15ms	remaining: 30ms
    2:	learn: 0.2986667	total: 21.7ms	remaining: 21.7ms
    3:	learn: 0.2482436	total: 35.8ms	remaining: 17.9ms
    4:	learn: 0.2052602	total: 43.8ms	remaining: 8.76ms
    5:	learn: 0.1699553	total: 51ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.5544651	total: 7.27ms	remaining: 43.6ms
    1:	learn: 0.4861574	total: 14.7ms	remaining: 36.8ms
    2:	learn: 0.4197661	total: 21.8ms	remaining: 29.1ms
    3:	learn: 0.3438917	total: 29.8ms	remaining: 22.4ms
    4:	learn: 0.2804893	total: 38.8ms	remaining: 15.5ms
    5:	learn: 0.2305684	total: 46.4ms	remaining: 7.74ms
    6:	learn: 0.2042735	total: 53.9ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.6196274	total: 6.76ms	remaining: 40.6ms
    1:	learn: 0.4893616	total: 13.7ms	remaining: 34.2ms
    2:	learn: 0.4129015	total: 23.7ms	remaining: 31.6ms
    3:	learn: 0.3524570	total: 34ms	remaining: 25.5ms
    4:	learn: 0.2932554	total: 42.3ms	remaining: 16.9ms
    5:	learn: 0.2446333	total: 50.7ms	remaining: 8.44ms
    6:	learn: 0.2174592	total: 57.5ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.5976336	total: 7.02ms	remaining: 42.1ms
    1:	learn: 0.4606599	total: 14.3ms	remaining: 35.7ms
    2:	learn: 0.3679934	total: 21.4ms	remaining: 28.5ms
    3:	learn: 0.3296789	total: 35ms	remaining: 26.2ms
    4:	learn: 0.2787401	total: 41.8ms	remaining: 16.7ms
    5:	learn: 0.2537074	total: 48.8ms	remaining: 8.13ms
    6:	learn: 0.2312155	total: 55.9ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.5712532	total: 6.96ms	remaining: 41.7ms
    1:	learn: 0.3989580	total: 14.3ms	remaining: 35.7ms
    2:	learn: 0.2999131	total: 21.5ms	remaining: 28.6ms
    3:	learn: 0.2556965	total: 28.6ms	remaining: 21.4ms
    4:	learn: 0.1987475	total: 43ms	remaining: 17.2ms
    5:	learn: 0.1832705	total: 52.4ms	remaining: 8.73ms
    6:	learn: 0.1804912	total: 59.5ms	remaining: 0us
    Learning rate set to 0.256882
    0:	learn: 0.5597579	total: 6.81ms	remaining: 40.8ms
    1:	learn: 0.4003889	total: 13.7ms	remaining: 34.2ms
    2:	learn: 0.3278358	total: 20.6ms	remaining: 27.5ms
    3:	learn: 0.2396006	total: 27.3ms	remaining: 20.5ms
    4:	learn: 0.1921558	total: 34ms	remaining: 13.6ms
    5:	learn: 0.1575824	total: 43.3ms	remaining: 7.22ms
    6:	learn: 0.1369980	total: 59.3ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.5676599	total: 7.34ms	remaining: 51.4ms
    1:	learn: 0.5028795	total: 14.5ms	remaining: 43.6ms
    2:	learn: 0.4385699	total: 21.6ms	remaining: 35.9ms
    3:	learn: 0.3671471	total: 28.7ms	remaining: 28.7ms
    4:	learn: 0.3052736	total: 35.8ms	remaining: 21.5ms
    5:	learn: 0.2550351	total: 42.9ms	remaining: 14.3ms
    6:	learn: 0.2280806	total: 50ms	remaining: 7.15ms
    7:	learn: 0.2122131	total: 57.1ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.6267272	total: 9.5ms	remaining: 66.5ms
    1:	learn: 0.5068829	total: 16.4ms	remaining: 49.3ms
    2:	learn: 0.4340781	total: 23.3ms	remaining: 38.8ms
    3:	learn: 0.3757349	total: 30.5ms	remaining: 30.5ms
    4:	learn: 0.3180081	total: 37.4ms	remaining: 22.5ms
    5:	learn: 0.2711385	total: 44.1ms	remaining: 14.7ms
    6:	learn: 0.2422556	total: 51.1ms	remaining: 7.29ms
    7:	learn: 0.2090404	total: 57.8ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.6062963	total: 13.8ms	remaining: 96.7ms
    1:	learn: 0.4802970	total: 21.9ms	remaining: 65.7ms
    2:	learn: 0.3918939	total: 32.8ms	remaining: 54.6ms
    3:	learn: 0.3543441	total: 40ms	remaining: 40ms
    4:	learn: 0.3055537	total: 47.3ms	remaining: 28.4ms
    5:	learn: 0.2806377	total: 54.4ms	remaining: 18.1ms
    6:	learn: 0.2573989	total: 61.8ms	remaining: 8.83ms
    7:	learn: 0.2250930	total: 69.3ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.5831077	total: 12.2ms	remaining: 85.5ms
    1:	learn: 0.4228500	total: 19.5ms	remaining: 58.6ms
    2:	learn: 0.3252042	total: 26.7ms	remaining: 44.5ms
    3:	learn: 0.2817922	total: 33.7ms	remaining: 33.7ms
    4:	learn: 0.2249644	total: 40.9ms	remaining: 24.5ms
    5:	learn: 0.2091171	total: 48.2ms	remaining: 16.1ms
    6:	learn: 0.2062568	total: 55.4ms	remaining: 7.92ms
    7:	learn: 0.1751816	total: 62.5ms	remaining: 0us
    Learning rate set to 0.227277
    0:	learn: 0.5723083	total: 6.9ms	remaining: 48.3ms
    1:	learn: 0.4227758	total: 13.9ms	remaining: 41.6ms
    2:	learn: 0.3614520	total: 21ms	remaining: 35ms
    3:	learn: 0.3023782	total: 27.9ms	remaining: 27.9ms
    4:	learn: 0.2485713	total: 35.1ms	remaining: 21.1ms
    5:	learn: 0.2115230	total: 42ms	remaining: 14ms
    6:	learn: 0.1933256	total: 48.9ms	remaining: 6.99ms
    7:	learn: 0.1458319	total: 55.8ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.5785225	total: 7.09ms	remaining: 56.7ms
    1:	learn: 0.5170965	total: 14.7ms	remaining: 51.6ms
    2:	learn: 0.4549549	total: 21.9ms	remaining: 43.8ms
    3:	learn: 0.3872806	total: 29ms	remaining: 36.3ms
    4:	learn: 0.3269564	total: 36.3ms	remaining: 29ms
    5:	learn: 0.2654394	total: 43.6ms	remaining: 21.8ms
    6:	learn: 0.2511984	total: 50.9ms	remaining: 14.5ms
    7:	learn: 0.2123315	total: 58.3ms	remaining: 7.29ms
    8:	learn: 0.1911696	total: 65.6ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.6325491	total: 11.1ms	remaining: 88.5ms
    1:	learn: 0.5215704	total: 18.5ms	remaining: 64.7ms
    2:	learn: 0.4522214	total: 25.8ms	remaining: 51.5ms
    3:	learn: 0.3958921	total: 33.2ms	remaining: 41.5ms
    4:	learn: 0.3397301	total: 40.4ms	remaining: 32.4ms
    5:	learn: 0.2945730	total: 48.4ms	remaining: 24.2ms
    6:	learn: 0.2645072	total: 55.8ms	remaining: 15.9ms
    7:	learn: 0.2303398	total: 62.9ms	remaining: 7.87ms
    8:	learn: 0.1970577	total: 70.3ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.6135026	total: 8.03ms	remaining: 64.3ms
    1:	learn: 0.4968027	total: 15.3ms	remaining: 53.4ms
    2:	learn: 0.4124687	total: 22.5ms	remaining: 44.9ms
    3:	learn: 0.3758127	total: 29.4ms	remaining: 36.8ms
    4:	learn: 0.3289506	total: 36.5ms	remaining: 29.2ms
    5:	learn: 0.3042852	total: 49.8ms	remaining: 24.9ms
    6:	learn: 0.2806333	total: 57.4ms	remaining: 16.4ms
    7:	learn: 0.2487638	total: 64.9ms	remaining: 8.12ms
    8:	learn: 0.2198546	total: 72.8ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.5928184	total: 7.15ms	remaining: 57.2ms
    1:	learn: 0.4431917	total: 14.4ms	remaining: 50.4ms
    2:	learn: 0.3475802	total: 21.7ms	remaining: 43.4ms
    3:	learn: 0.3051114	total: 28.9ms	remaining: 36.1ms
    4:	learn: 0.2486775	total: 36.4ms	remaining: 29.1ms
    5:	learn: 0.2205133	total: 43.4ms	remaining: 21.7ms
    6:	learn: 0.2102176	total: 50.5ms	remaining: 14.4ms
    7:	learn: 0.1838189	total: 58.3ms	remaining: 7.28ms
    8:	learn: 0.1564763	total: 64.9ms	remaining: 0us
    Learning rate set to 0.204008
    0:	learn: 0.5826713	total: 6.95ms	remaining: 55.6ms
    1:	learn: 0.4420531	total: 14ms	remaining: 49.2ms
    2:	learn: 0.3818730	total: 21ms	remaining: 42.1ms
    3:	learn: 0.3257271	total: 28.1ms	remaining: 35.1ms
    4:	learn: 0.2720370	total: 34.8ms	remaining: 27.8ms
    5:	learn: 0.2342646	total: 41.5ms	remaining: 20.8ms
    6:	learn: 0.2148025	total: 48.3ms	remaining: 13.8ms
    7:	learn: 0.1643983	total: 55.2ms	remaining: 6.9ms
    8:	learn: 0.1394023	total: 62.2ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.6226827	total: 6.96ms	remaining: 41.8ms
    1:	learn: 0.5769698	total: 14.2ms	remaining: 35.5ms
    2:	learn: 0.5119934	total: 21.3ms	remaining: 28.4ms
    3:	learn: 0.4678750	total: 28.4ms	remaining: 21.3ms
    4:	learn: 0.4483644	total: 35.8ms	remaining: 14.3ms
    5:	learn: 0.4291777	total: 43.3ms	remaining: 7.22ms
    6:	learn: 0.3790653	total: 50.6ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.6275101	total: 7.17ms	remaining: 43ms
    1:	learn: 0.5307171	total: 14.5ms	remaining: 36.4ms
    2:	learn: 0.4758290	total: 21.9ms	remaining: 29.2ms
    3:	learn: 0.4459674	total: 29.2ms	remaining: 21.9ms
    4:	learn: 0.3681349	total: 36.5ms	remaining: 14.6ms
    5:	learn: 0.3510063	total: 43.8ms	remaining: 7.31ms
    6:	learn: 0.3298754	total: 51.2ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.5723992	total: 7.46ms	remaining: 44.8ms
    1:	learn: 0.5147729	total: 15ms	remaining: 37.4ms
    2:	learn: 0.4773892	total: 22.5ms	remaining: 30ms
    3:	learn: 0.4185524	total: 30.3ms	remaining: 22.7ms
    4:	learn: 0.3740797	total: 38.3ms	remaining: 15.3ms
    5:	learn: 0.3518636	total: 48.9ms	remaining: 8.15ms
    6:	learn: 0.3384991	total: 56.6ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.5944673	total: 7.25ms	remaining: 43.5ms
    1:	learn: 0.5144252	total: 14.7ms	remaining: 36.8ms
    2:	learn: 0.4390108	total: 22ms	remaining: 29.3ms
    3:	learn: 0.3876054	total: 29.4ms	remaining: 22ms
    4:	learn: 0.3660083	total: 36.7ms	remaining: 14.7ms
    5:	learn: 0.3406763	total: 43.8ms	remaining: 7.3ms
    6:	learn: 0.3281843	total: 51.1ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.6335941	total: 7.37ms	remaining: 44.2ms
    1:	learn: 0.5566898	total: 15.6ms	remaining: 39ms
    2:	learn: 0.5158836	total: 23.5ms	remaining: 31.3ms
    3:	learn: 0.4635754	total: 31.5ms	remaining: 23.6ms
    4:	learn: 0.4368181	total: 39.3ms	remaining: 15.7ms
    5:	learn: 0.4041337	total: 47.3ms	remaining: 7.89ms
    6:	learn: 0.3535831	total: 55.3ms	remaining: 0us
    Learning rate set to 0.365657
    0:	learn: 0.5876968	total: 15.6ms	remaining: 109ms
    1:	learn: 0.5238632	total: 30.6ms	remaining: 91.9ms
    2:	learn: 0.4506242	total: 45.6ms	remaining: 76.1ms
    3:	learn: 0.4205261	total: 60.5ms	remaining: 60.5ms
    4:	learn: 0.3608865	total: 75.8ms	remaining: 45.5ms
    5:	learn: 0.3182687	total: 91.3ms	remaining: 30.4ms
    6:	learn: 0.2674556	total: 109ms	remaining: 15.6ms
    7:	learn: 0.2574000	total: 124ms	remaining: 0us
    Learning rate set to 0.365657
    0:	learn: 0.5597653	total: 15.3ms	remaining: 107ms
    1:	learn: 0.4928334	total: 30.9ms	remaining: 92.7ms
    2:	learn: 0.4196772	total: 46.2ms	remaining: 77ms
    3:	learn: 0.3645659	total: 61.4ms	remaining: 61.4ms
    4:	learn: 0.3258771	total: 76.9ms	remaining: 46.1ms
    5:	learn: 0.2823132	total: 94.2ms	remaining: 31.4ms
    6:	learn: 0.2611093	total: 109ms	remaining: 15.6ms
    7:	learn: 0.2407104	total: 125ms	remaining: 0us
    Learning rate set to 0.365657
    0:	learn: 0.5648633	total: 15.6ms	remaining: 110ms
    1:	learn: 0.4795053	total: 31.3ms	remaining: 93.9ms
    2:	learn: 0.4086578	total: 46.7ms	remaining: 77.9ms
    3:	learn: 0.3478656	total: 62.2ms	remaining: 62.2ms
    4:	learn: 0.3024778	total: 77.7ms	remaining: 46.6ms
    5:	learn: 0.2822529	total: 93.3ms	remaining: 31.1ms
    6:	learn: 0.2628832	total: 109ms	remaining: 15.6ms
    7:	learn: 0.2454696	total: 125ms	remaining: 0us
    Learning rate set to 0.365657
    0:	learn: 0.5148529	total: 15.2ms	remaining: 106ms
    1:	learn: 0.4351507	total: 30.7ms	remaining: 92.2ms
    2:	learn: 0.3820530	total: 49.4ms	remaining: 82.3ms
    3:	learn: 0.3370742	total: 66.4ms	remaining: 66.4ms
    4:	learn: 0.3154005	total: 82.1ms	remaining: 49.3ms
    5:	learn: 0.2695623	total: 98.1ms	remaining: 32.7ms
    6:	learn: 0.2397894	total: 114ms	remaining: 16.3ms
    7:	learn: 0.2149842	total: 130ms	remaining: 0us
    Learning rate set to 0.365657
    0:	learn: 0.5777914	total: 16.8ms	remaining: 118ms
    1:	learn: 0.5213328	total: 32.3ms	remaining: 97ms
    2:	learn: 0.4651652	total: 49.2ms	remaining: 82.1ms
    3:	learn: 0.4253230	total: 72.5ms	remaining: 72.5ms
    4:	learn: 0.3594876	total: 94.5ms	remaining: 56.7ms
    5:	learn: 0.3254442	total: 110ms	remaining: 36.6ms
    6:	learn: 0.3161210	total: 126ms	remaining: 17.9ms
    7:	learn: 0.2685397	total: 141ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5625499	total: 17.8ms	remaining: 53.5ms
    1:	learn: 0.4911557	total: 33.3ms	remaining: 33.3ms
    2:	learn: 0.3931344	total: 48.5ms	remaining: 16.2ms
    3:	learn: 0.3314958	total: 63.7ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5307313	total: 18.4ms	remaining: 55.1ms
    1:	learn: 0.4567318	total: 33.7ms	remaining: 33.7ms
    2:	learn: 0.3778222	total: 49ms	remaining: 16.3ms
    3:	learn: 0.3533667	total: 67.4ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5356930	total: 15.3ms	remaining: 46ms
    1:	learn: 0.4543764	total: 30.9ms	remaining: 30.9ms
    2:	learn: 0.3700968	total: 46.2ms	remaining: 15.4ms
    3:	learn: 0.2980827	total: 61.8ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.4800252	total: 15ms	remaining: 44.9ms
    1:	learn: 0.3325932	total: 30ms	remaining: 30ms
    2:	learn: 0.2785551	total: 44.9ms	remaining: 15ms
    3:	learn: 0.2514924	total: 68.5ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5522850	total: 15.3ms	remaining: 45.9ms
    1:	learn: 0.4869528	total: 35.9ms	remaining: 35.9ms
    2:	learn: 0.4194038	total: 51.7ms	remaining: 17.2ms
    3:	learn: 0.3836124	total: 67ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.5779947	total: 14.7ms	remaining: 88.4ms
    1:	learn: 0.5109754	total: 29.7ms	remaining: 74.2ms
    2:	learn: 0.4276611	total: 45ms	remaining: 60ms
    3:	learn: 0.3672238	total: 60.1ms	remaining: 45ms
    4:	learn: 0.3161149	total: 75.4ms	remaining: 30.2ms
    5:	learn: 0.2826077	total: 90.7ms	remaining: 15.1ms
    6:	learn: 0.2639800	total: 105ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.5483555	total: 18.1ms	remaining: 109ms
    1:	learn: 0.4783521	total: 32.8ms	remaining: 82ms
    2:	learn: 0.4016744	total: 47.6ms	remaining: 63.5ms
    3:	learn: 0.3444224	total: 62.6ms	remaining: 47ms
    4:	learn: 0.3160606	total: 78ms	remaining: 31.2ms
    5:	learn: 0.2700949	total: 93.2ms	remaining: 15.5ms
    6:	learn: 0.2458242	total: 108ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.5535080	total: 15.5ms	remaining: 93ms
    1:	learn: 0.4653713	total: 31.4ms	remaining: 78.4ms
    2:	learn: 0.4065343	total: 47.7ms	remaining: 63.6ms
    3:	learn: 0.3433595	total: 66.9ms	remaining: 50.1ms
    4:	learn: 0.2922735	total: 82.4ms	remaining: 33ms
    5:	learn: 0.2760245	total: 98.1ms	remaining: 16.3ms
    6:	learn: 0.2506992	total: 115ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.5010208	total: 15.3ms	remaining: 91.6ms
    1:	learn: 0.3493941	total: 30.7ms	remaining: 76.8ms
    2:	learn: 0.2972360	total: 49.5ms	remaining: 66ms
    3:	learn: 0.2579114	total: 65.2ms	remaining: 48.9ms
    4:	learn: 0.2229195	total: 81ms	remaining: 32.4ms
    5:	learn: 0.1873861	total: 96.5ms	remaining: 16.1ms
    6:	learn: 0.1612720	total: 112ms	remaining: 0us
    Learning rate set to 0.413288
    0:	learn: 0.5678085	total: 15.3ms	remaining: 91.9ms
    1:	learn: 0.5056291	total: 30.8ms	remaining: 77.1ms
    2:	learn: 0.4397573	total: 46.7ms	remaining: 62.3ms
    3:	learn: 0.4086162	total: 63.5ms	remaining: 47.6ms
    4:	learn: 0.3724886	total: 78.4ms	remaining: 31.4ms
    5:	learn: 0.3285586	total: 93.4ms	remaining: 15.6ms
    6:	learn: 0.2737405	total: 109ms	remaining: 0us
    Learning rate set to 0.328222
    0:	learn: 0.5959620	total: 14.4ms	remaining: 115ms
    1:	learn: 0.5351239	total: 34.5ms	remaining: 121ms
    2:	learn: 0.4657388	total: 49.4ms	remaining: 98.8ms
    3:	learn: 0.4372422	total: 64.6ms	remaining: 80.7ms
    4:	learn: 0.4051824	total: 79.4ms	remaining: 63.5ms
    5:	learn: 0.3682002	total: 94.3ms	remaining: 47.2ms
    6:	learn: 0.3371953	total: 109ms	remaining: 31.2ms
    7:	learn: 0.3114377	total: 124ms	remaining: 15.5ms
    8:	learn: 0.2898687	total: 138ms	remaining: 0us
    Learning rate set to 0.328222
    0:	learn: 0.5696235	total: 16.2ms	remaining: 129ms
    1:	learn: 0.5056488	total: 31ms	remaining: 109ms
    2:	learn: 0.4356585	total: 46.4ms	remaining: 92.8ms
    3:	learn: 0.3824667	total: 61.6ms	remaining: 77ms
    4:	learn: 0.3442138	total: 76.7ms	remaining: 61.4ms
    5:	learn: 0.3167226	total: 92ms	remaining: 46ms
    6:	learn: 0.2941679	total: 107ms	remaining: 30.7ms
    7:	learn: 0.2656069	total: 123ms	remaining: 15.4ms
    8:	learn: 0.2443185	total: 139ms	remaining: 0us
    Learning rate set to 0.328222
    0:	learn: 0.5746044	total: 15.6ms	remaining: 125ms
    1:	learn: 0.4866135	total: 31.2ms	remaining: 109ms
    2:	learn: 0.4133516	total: 47ms	remaining: 94ms
    3:	learn: 0.3619792	total: 63ms	remaining: 78.8ms
    4:	learn: 0.3224831	total: 78.7ms	remaining: 63ms
    5:	learn: 0.3043770	total: 94.4ms	remaining: 47.2ms
    6:	learn: 0.2849875	total: 111ms	remaining: 31.7ms
    7:	learn: 0.2624387	total: 127ms	remaining: 15.9ms
    8:	learn: 0.2482090	total: 143ms	remaining: 0us
    Learning rate set to 0.328222
    0:	learn: 0.5270035	total: 15.4ms	remaining: 123ms
    1:	learn: 0.4336282	total: 33.9ms	remaining: 119ms
    2:	learn: 0.3771420	total: 57ms	remaining: 114ms
    3:	learn: 0.3453120	total: 72.2ms	remaining: 90.3ms
    4:	learn: 0.3005976	total: 87.5ms	remaining: 70ms
    5:	learn: 0.2527591	total: 103ms	remaining: 51.6ms
    6:	learn: 0.2253157	total: 119ms	remaining: 33.9ms
    7:	learn: 0.2206671	total: 134ms	remaining: 16.8ms
    8:	learn: 0.1961774	total: 150ms	remaining: 0us
    Learning rate set to 0.328222
    0:	learn: 0.5864005	total: 15.4ms	remaining: 123ms
    1:	learn: 0.5349851	total: 31.3ms	remaining: 110ms
    2:	learn: 0.4864683	total: 47ms	remaining: 94ms
    3:	learn: 0.4502974	total: 62.7ms	remaining: 78.3ms
    4:	learn: 0.4145949	total: 78.3ms	remaining: 62.6ms
    5:	learn: 0.3820392	total: 94ms	remaining: 47ms
    6:	learn: 0.3467805	total: 110ms	remaining: 31.3ms
    7:	learn: 0.2938968	total: 125ms	remaining: 15.6ms
    8:	learn: 0.2670494	total: 140ms	remaining: 0us
    Learning rate set to 0.47604
    0:	learn: 0.5665352	total: 14.4ms	remaining: 72.1ms
    1:	learn: 0.4961987	total: 28.8ms	remaining: 57.6ms
    2:	learn: 0.4003102	total: 43.2ms	remaining: 43.2ms
    3:	learn: 0.3401025	total: 57.6ms	remaining: 28.8ms
    4:	learn: 0.2959208	total: 72.5ms	remaining: 14.5ms
    5:	learn: 0.2646336	total: 87.4ms	remaining: 0us
    Learning rate set to 0.47604
    0:	learn: 0.5352202	total: 15.1ms	remaining: 75.4ms
    1:	learn: 0.4620923	total: 31.1ms	remaining: 62.1ms
    2:	learn: 0.3814787	total: 46.5ms	remaining: 46.5ms
    3:	learn: 0.3256727	total: 61.8ms	remaining: 30.9ms
    4:	learn: 0.2761112	total: 77.3ms	remaining: 15.5ms
    5:	learn: 0.2410166	total: 92.8ms	remaining: 0us
    Learning rate set to 0.47604
    0:	learn: 0.5402571	total: 15.9ms	remaining: 79.7ms
    1:	learn: 0.4593915	total: 32ms	remaining: 63.9ms
    2:	learn: 0.3761994	total: 47.9ms	remaining: 47.9ms
    3:	learn: 0.3052379	total: 64.1ms	remaining: 32ms
    4:	learn: 0.2613503	total: 79.9ms	remaining: 16ms
    5:	learn: 0.2411308	total: 95.9ms	remaining: 0us
    Learning rate set to 0.47604
    0:	learn: 0.4853287	total: 15.5ms	remaining: 77.7ms
    1:	learn: 0.3406781	total: 33.8ms	remaining: 67.6ms
    2:	learn: 0.2869595	total: 49.2ms	remaining: 49.2ms
    3:	learn: 0.2609136	total: 64.5ms	remaining: 32.3ms
    4:	learn: 0.2351232	total: 79.9ms	remaining: 16ms
    5:	learn: 0.1867712	total: 95.6ms	remaining: 0us
    Learning rate set to 0.47604
    0:	learn: 0.5562427	total: 15.3ms	remaining: 76.4ms
    1:	learn: 0.4916582	total: 31ms	remaining: 61.9ms
    2:	learn: 0.4244436	total: 46.9ms	remaining: 46.9ms
    3:	learn: 0.3899496	total: 62.9ms	remaining: 31.4ms
    4:	learn: 0.3539591	total: 78.5ms	remaining: 15.7ms
    5:	learn: 0.3054075	total: 94ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5664496	total: 30.1ms	remaining: 211ms
    1:	learn: 0.4653989	total: 60.2ms	remaining: 181ms
    2:	learn: 0.4306863	total: 90.4ms	remaining: 151ms
    3:	learn: 0.3808484	total: 123ms	remaining: 123ms
    4:	learn: 0.3268862	total: 156ms	remaining: 93.3ms
    5:	learn: 0.2782603	total: 185ms	remaining: 61.8ms
    6:	learn: 0.2721710	total: 223ms	remaining: 31.9ms
    7:	learn: 0.2346461	total: 254ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5667362	total: 29.6ms	remaining: 207ms
    1:	learn: 0.4810802	total: 59.3ms	remaining: 178ms
    2:	learn: 0.4148173	total: 91.9ms	remaining: 153ms
    3:	learn: 0.3645654	total: 121ms	remaining: 121ms
    4:	learn: 0.3306388	total: 152ms	remaining: 90.9ms
    5:	learn: 0.2949094	total: 181ms	remaining: 60.3ms
    6:	learn: 0.2666643	total: 216ms	remaining: 30.8ms
    7:	learn: 0.2478438	total: 249ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5420385	total: 32.6ms	remaining: 228ms
    1:	learn: 0.4489425	total: 62.5ms	remaining: 187ms
    2:	learn: 0.3623975	total: 92.3ms	remaining: 154ms
    3:	learn: 0.3240729	total: 122ms	remaining: 122ms
    4:	learn: 0.2798172	total: 152ms	remaining: 91.4ms
    5:	learn: 0.2485001	total: 182ms	remaining: 60.8ms
    6:	learn: 0.2297927	total: 223ms	remaining: 31.8ms
    7:	learn: 0.2034911	total: 254ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5719104	total: 29.3ms	remaining: 205ms
    1:	learn: 0.4959806	total: 59.1ms	remaining: 177ms
    2:	learn: 0.4104861	total: 88.8ms	remaining: 148ms
    3:	learn: 0.3560124	total: 123ms	remaining: 123ms
    4:	learn: 0.3232127	total: 153ms	remaining: 91.8ms
    5:	learn: 0.2826410	total: 183ms	remaining: 60.9ms
    6:	learn: 0.2552242	total: 224ms	remaining: 32ms
    7:	learn: 0.2394226	total: 254ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5620640	total: 32.1ms	remaining: 225ms
    1:	learn: 0.4676063	total: 62.1ms	remaining: 186ms
    2:	learn: 0.4033658	total: 92.3ms	remaining: 154ms
    3:	learn: 0.3473098	total: 125ms	remaining: 125ms
    4:	learn: 0.3175938	total: 156ms	remaining: 93.3ms
    5:	learn: 0.2853292	total: 186ms	remaining: 61.9ms
    6:	learn: 0.2659830	total: 226ms	remaining: 32.2ms
    7:	learn: 0.2292919	total: 256ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5664496	total: 29ms	remaining: 232ms
    1:	learn: 0.4653989	total: 59.3ms	remaining: 207ms
    2:	learn: 0.4306863	total: 88.5ms	remaining: 177ms
    3:	learn: 0.3808484	total: 118ms	remaining: 148ms
    4:	learn: 0.3268862	total: 148ms	remaining: 119ms
    5:	learn: 0.2782603	total: 178ms	remaining: 88.9ms
    6:	learn: 0.2721710	total: 213ms	remaining: 60.8ms
    7:	learn: 0.2346461	total: 260ms	remaining: 32.5ms
    8:	learn: 0.2118896	total: 290ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5667362	total: 32.5ms	remaining: 260ms
    1:	learn: 0.4810802	total: 62.2ms	remaining: 218ms
    2:	learn: 0.4148173	total: 91.6ms	remaining: 183ms
    3:	learn: 0.3645654	total: 122ms	remaining: 152ms
    4:	learn: 0.3306388	total: 151ms	remaining: 121ms
    5:	learn: 0.2949094	total: 182ms	remaining: 90.9ms
    6:	learn: 0.2666643	total: 220ms	remaining: 62.9ms
    7:	learn: 0.2478438	total: 250ms	remaining: 31.3ms
    8:	learn: 0.2168089	total: 280ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5420385	total: 29.3ms	remaining: 234ms
    1:	learn: 0.4489425	total: 58.5ms	remaining: 205ms
    2:	learn: 0.3623975	total: 88.3ms	remaining: 177ms
    3:	learn: 0.3240729	total: 121ms	remaining: 151ms
    4:	learn: 0.2798172	total: 151ms	remaining: 121ms
    5:	learn: 0.2485001	total: 180ms	remaining: 90ms
    6:	learn: 0.2297927	total: 217ms	remaining: 61.9ms
    7:	learn: 0.2034911	total: 255ms	remaining: 31.8ms
    8:	learn: 0.1819136	total: 287ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5719104	total: 29.3ms	remaining: 235ms
    1:	learn: 0.4959806	total: 58.6ms	remaining: 205ms
    2:	learn: 0.4104861	total: 91.6ms	remaining: 183ms
    3:	learn: 0.3560124	total: 122ms	remaining: 152ms
    4:	learn: 0.3232127	total: 153ms	remaining: 122ms
    5:	learn: 0.2826410	total: 183ms	remaining: 91.6ms
    6:	learn: 0.2552242	total: 219ms	remaining: 62.5ms
    7:	learn: 0.2394226	total: 249ms	remaining: 31.1ms
    8:	learn: 0.2141019	total: 279ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5620640	total: 29.4ms	remaining: 235ms
    1:	learn: 0.4676063	total: 59.7ms	remaining: 209ms
    2:	learn: 0.4033658	total: 89.6ms	remaining: 179ms
    3:	learn: 0.3473098	total: 119ms	remaining: 149ms
    4:	learn: 0.3175938	total: 149ms	remaining: 119ms
    5:	learn: 0.2853292	total: 179ms	remaining: 89.5ms
    6:	learn: 0.2659830	total: 232ms	remaining: 66.2ms
    7:	learn: 0.2292919	total: 262ms	remaining: 32.8ms
    8:	learn: 0.2026368	total: 295ms	remaining: 0us
    Learning rate set to 0.5
    0:	learn: 0.5344689	total: 32.6ms	remaining: 260ms
    1:	learn: 0.4561615	total: 64.2ms	remaining: 225ms
    2:	learn: 0.3971815	total: 95.6ms	remaining: 191ms
    3:	learn: 0.3409930	total: 126ms	remaining: 157ms
    4:	learn: 0.2905572	total: 156ms	remaining: 125ms
    5:	learn: 0.2695671	total: 186ms	remaining: 93ms
    6:	learn: 0.2465751	total: 223ms	remaining: 63.7ms
    7:	learn: 0.2244980	total: 254ms	remaining: 31.8ms
    8:	learn: 0.2039606	total: 284ms	remaining: 0us
    CPU times: user 6min 29s, sys: 1min 15s, total: 7min 45s
    Wall time: 5min 19s
    


```python
print("Лучший множитель:", best_xN)
print("Лучший f1_score:", round(best_F1, 2))
print("Лучшая модель:", best_model)
```

    Лучший множитель: 4
    Лучший f1_score: 0.86
    Лучшая модель: RandomForestClassifier(max_depth=8, n_estimators=25)
    


```python
best_model_on_data_custom = best_model
```

### Вывод

**В результате поиска множителя балансировки, а также подбора лучших параметров моделей на custom данных, установлено:**

1. Высокое качество модели RandomForestClassifier(max_depth=8, n_estimators=25) достигается на множителе - 4.
2. Значение F1: 0.86.
3. Время поиска множителя для балансировки и параметров модели порядка: 8 мин.

## Проверка качества модели на тестовой выборке

### best_model_on_data_default


```python
best_model_on_data_default.fit(features_train, target_train)
test_predictions = best_model_on_data_default.predict(features_test)
result = f1_score(target_test, test_predictions)
print("F1 лучшей модели", best_model_on_data_default, "на тестовой выборке:", abs(round(result, 2)))
```

    F1 лучшей модели LogisticRegression(C=18, class_weight='balanced') на тестовой выборке: 0.52
    

### best_model_on_data_custom


```python
features = pd.DataFrame(features)
target = pd.DataFrame(df['toxic'])
```


```python
dat = np.concatenate((features, target), axis=1)
dat = pd.DataFrame(dat)
dat = shuffle(
        dat, random_state=23031998)

```


```python
dat_up = upsample(dat, 4)
```


```python
features_train_up, features_test_up, target_train_up, target_test_up = train_test_split(dat_up.iloc[:,:-1], dat_up.iloc[:,-1], test_size=0.25)
```


```python
best_model_on_data_custom.fit(features_train_up, target_train_up)
test_predictions = best_model_on_data_custom.predict(features_test_up)
result = f1_score(target_test_up, test_predictions)
print("F1 лучшей модели", best_model_on_data_custom, "на тестовой выборке:", abs(round(result, 2)))
```

    F1 лучшей модели RandomForestClassifier(max_depth=8, n_estimators=25) на тестовой выборке: 0.96
    

# Общий вывод

**I. В результате загрузки данных, установлено:**

1.  DataFrame содержит 159571 строк и 2 столбца.
2.  В столбце «text» данные типа object, пропуски отсутствуют.
3.  В столбце «toxic» данные типа int64, пропуски отсутствуют.
4.  Явные дубликаты отсутствуют.

**II. В результате предобработки:**

1. Произведена очистка данных и лемматизация (для BERT можно и не делать лемматизацию, она и так хорошо справляется).
2. Выбрано случайным образом 500 строк, ввиду ресурсоёмкости обработки всего массива данных.
3. Установлено, что соотношение класса '0' и '1' соответственно: 0.88 : 0.12, данный факт указывает на необходимость балансировки, однако, для начала необходимо подготовить модели.

**III. В результате подготовки модели, а также features & target:**

1.  Загружена предобученная модель `BertModel`.
2.  Выполнена токенизация и кодирование строк.
3.  Выполнена подрезка длинны токенов (не должно превышать 512 (обусловлено особенностями используемой модели BERT) 
4.  Выборка разделена на тренировочную и тестовую (75:25).

**IV. В результате обучения и подбора лучших параметров моделей на default данных, установлено:**

1. Лучшая модель: LogisticRegression.
2. Параметры лучшей модели: C=18, class_weight='balanced'.
3. Качество модели (F1): 0.444.
4. Не высокое качество модели обусловлено дисбалансом классов.

**V. В результате поиска множителя балансировки, а также подбора лучших параметров моделей на custom данных, установлено:**

1. Высокое качество модели RandomForestClassifier(max_depth=8, n_estimators=25) достигается на множителе - 4.
2. Значение F1: 0.86.
3. Время поиска множителя для балансировки и параметров модели порядка: 8 мин.

**VI. В результате проверки качества модели на тестовой выборке, установлено:**

1. для best_model_on_data_default
  * F1 лучшей модели LogisticRegression(C=18, class_weight='balanced') на тестовой выборке: 0.52;
  * Требование бизнеса (F1 не меньше 0.75) – не выполнено;
2. для best_model_on_data_custom
  * F1 лучшей модели RandomForestClassifier(max_depth=8, n_estimators=25) на тестовой выборке: 0.96;
  * Множитель, увеличивающий выборку - 4;
  * Требование бизнеса (F1 не меньше 0.75) - выполнено.


```python

```
