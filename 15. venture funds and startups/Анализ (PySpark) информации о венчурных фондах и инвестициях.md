## Загрузка библиотек, данных


```python
!pip install -U -q PyDrive
!pip install pyspark
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: pyspark in /usr/local/lib/python3.7/dist-packages (3.3.0)
    Requirement already satisfied: py4j==0.10.9.5 in /usr/local/lib/python3.7/dist-packages (from pyspark) (0.10.9.5)
    


```python
import numpy as np
import pandas as pd

from google.colab import drive
from google.colab import files
import os
```


```python
from pyspark.sql import SparkSession
spark = SparkSession \
.builder \
.appName("Python Spark SQL basic example") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()
```


```python
# просмотр, где находится каталог с файлами на COLAB
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
# получение доступа к каталогу и уточнение названия папок
os.listdir('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ')
```




    ['acquisitions.csv',
     'degrees.csv',
     'funds.csv',
     'investments.csv',
     'funding_rounds.csv',
     'ipos.csv',
     'milestones.csv',
     'objects.csv',
     'offices.csv',
     'people.csv',
     'relationships.csv']




```python
# загрузка данных
acquisition = spark.read.load('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ/acquisitions.csv', 
                       format='csv', header='true', inferSchema='true').createOrReplaceTempView('acquisition')
education = spark.read.load('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ/degrees.csv', 
                       format='csv', header='true', inferSchema='true').createOrReplaceTempView('education')
fund = spark.read.load('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ/funds.csv', 
                       format='csv', header='true', inferSchema='true').createOrReplaceTempView('fund')
investment = spark.read.load('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ/investments.csv', 
                       format='csv', header='true', inferSchema='true').createOrReplaceTempView('investment')
funding_round = spark.read.load('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ/funding_rounds.csv', 
                       format='csv', header='true', inferSchema='true').createOrReplaceTempView('funding_round')
company = spark.read.load('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ/objects.csv', 
                       format='csv', header='true', inferSchema='true').createOrReplaceTempView('company')
people = spark.read.load('/content/drive/My Drive/Colab Notebooks/Яндекс/Проект 15/ДАННЫЕ/people.csv', 
                       format='csv', header='true', inferSchema='true').createOrReplaceTempView('people')
```

## SQL - запросы

### 1. Количество компаний, которые закрылись.


```python
spark.sql(
    "SELECT COUNT(id)\
    FROM company\
    WHERE status = 'closed'"
    ).show()
```

    +---------+
    |count(id)|
    +---------+
    |     2773|
    +---------+
    
    

### 2. Количество привлечённых средств для новостных компаний США.


```python
spark.sql(
    "SELECT funding_total_usd FROM company\
    WHERE category_code LIKE '%new%'\
    AND country_code = 'USA'\
    ORDER BY funding_total_usd DESC"
    ).show()
```

    +-----------------+
    |funding_total_usd|
    +-----------------+
    |        9500000.0|
    |        8428250.0|
    |         828000.0|
    |        7704308.0|
    |        7380000.0|
    |           7100.0|
    |        7000000.0|
    |       69000000.0|
    |        6500000.0|
    |        6250000.0|
    |      622552813.0|
    |       56464869.0|
    |         562263.0|
    |         560000.0|
    |         540000.0|
    |         539999.0|
    |        5000000.0|
    |        4750000.0|
    |       46300000.0|
    |        4575000.0|
    +-----------------+
    only showing top 20 rows
    
    

### 3. Общая сумма сделок по покупке одних компаний другими (сделки, которые осуществлялись только за наличные с 2011 по 2013 год включительно).


```python
spark.sql(
    "SELECT SUM(price_amount)\
    FROM acquisition\
    WHERE (CAST(acquired_at AS date) BETWEEN '2011-01-01' AND '2013-12-31')\
    AND term_code = 'cash'"
    ).show()
```

    +-----------------+
    |sum(price_amount)|
    +-----------------+
    | 1.39077174636E11|
    +-----------------+
    
    

### 4. Имя, фамилия и название аккаунтов людей в твиттере, у которых названия аккаунтов начинаются на 'Silver'.


```python
spark.sql(
    "SELECT first_name,\
    last_name,\
    twitter_username\
    FROM people\
    JOIN company ON company.id=people.object_id\
    WHERE twitter_username LIKE 'Silver%'"
    ).show()
```

    +----------+---------+----------------+
    |first_name|last_name|twitter_username|
    +----------+---------+----------------+
    |   Rebecca|   Silver|   SilverRebecca|
    |    Silver|    Teede|   SilverMatrixx|
    |   Mattias| Guilotte|     Silverreven|
    +----------+---------+----------------+
    
    

### 5. Информация о людях, у которых названия аккаунтов в твиттере содержат подстроку 'money', а фамилия начинается на 'K'.


```python
spark.sql(
    "SELECT first_name,\
    last_name,\
    twitter_username\
    FROM people\
    JOIN company ON company.id=people.object_id\
    WHERE twitter_username LIKE '%money%'\
    AND last_name LIKE 'K%'"
    ).show()
```

    +----------+---------+----------------+
    |first_name|last_name|twitter_username|
    +----------+---------+----------------+
    |   Gregory|      Kim|        gmoney75|
    +----------+---------+----------------+
    
    

### 6. Общая сумма привлечённых инвестиций, которые получили компании, зарегистрированные в стране.


```python
spark.sql(
    "SELECT country_code,\
    SUM(funding_total_usd)\
    FROM company\
    GROUP BY country_code\
    ORDER BY SUM(funding_total_usd) DESC"
    ).show()
```

    +------------+----------------------+
    |country_code|sum(funding_total_usd)|
    +------------+----------------------+
    |         USA|      1.57520038925E11|
    |         GBR|         8.326609459E9|
    |        null|         8.079516097E9|
    |         CHN|         4.833281629E9|
    |         CAN|         4.684222749E9|
    |         DEU|         2.191433228E9|
    |         FRA|         1.996523096E9|
    |         ISR|         1.936009731E9|
    |         CHE|         1.918123682E9|
    |         IND|         1.853089881E9|
    |         JPN|         1.300027697E9|
    |         NLD|         1.087397382E9|
    |         DNK|          8.23043845E8|
    |         FIN|          7.33672508E8|
    |         AUS|          5.95965806E8|
    |         BRA|          5.59674625E8|
    |         ESP|          5.59531025E8|
    |         BEL|          5.16780482E8|
    |         SWE|          5.05728093E8|
    |         RUS|            4.639875E8|
    +------------+----------------------+
    only showing top 20 rows
    
    

### 7. Дата проведения раунда, а также минимальное и максимальное значения суммы инвестиций, привлечённых в эту дату (в итоговой таблице только те записи, в которых минимальное значение суммы инвестиций не равно нулю и не равно максимальному значению).


```python
spark.sql(
    "SELECT funded_at,\
    MIN(raised_amount),\
    MAX(raised_amount)\
    FROM funding_round\
    GROUP BY funded_at\
    HAVING MIN(raised_amount) != 0\
    AND MIN(raised_amount) != MAX(raised_amount)"
    ).show()
```

    +-------------------+------------------+------------------+
    |          funded_at|min(raised_amount)|max(raised_amount)|
    +-------------------+------------------+------------------+
    |2003-02-01 00:00:00|          190000.0|             4.1E7|
    |2008-03-08 00:00:00|          800000.0|         7000000.0|
    |2006-12-21 00:00:00|         2700000.0|         5300000.0|
    |2005-12-15 00:00:00|         3800000.0|             1.0E7|
    |2005-11-16 00:00:00|         1260000.0|             2.0E7|
    |2006-12-06 00:00:00|          650000.0|            1.45E7|
    |2007-07-24 00:00:00|          500000.0|             1.5E7|
    |2009-06-10 00:00:00|          230000.0|             2.4E7|
    |2006-05-25 00:00:00|          500000.0|             1.2E7|
    |2012-10-20 00:00:00|           40000.0|         4700000.0|
    |2005-04-26 00:00:00|         1300000.0|             2.0E7|
    |2010-05-03 00:00:00|           32723.0|            3.88E7|
    |2005-04-13 00:00:00|         1940000.0|         7990000.0|
    |2010-11-02 00:00:00|           50000.0|             2.2E7|
    |2012-01-22 00:00:00|           48501.0|         1100000.0|
    |2003-06-01 00:00:00|          250000.0|            1.95E7|
    |2007-03-28 00:00:00|          209000.0|             6.0E7|
    |2010-06-26 00:00:00|           30000.0|         3000000.0|
    |2009-11-27 00:00:00|          100000.0|             1.8E8|
    |2010-05-07 00:00:00|          100000.0|             3.0E7|
    +-------------------+------------------+------------------+
    only showing top 20 rows
    
    

### 8. Поле с категориями:
* для фондов, которые инвестируют в 100 и более компаний - high_activity;
* для фондов, которые инвестируют в 20 и более компаний до 100 - middle_activity;
* если количество инвестируемых компаний фонда не достигает 20 - low_activity.

(отображены поле name таблицы fund и поле с категориями)



```python
spark.sql(
    "SELECT fund.name,\
    CASE\
    WHEN company.invested_companies >= 100 THEN 'high_activity'\
    WHEN company.invested_companies  >= 20 THEN 'middle_activity'\
    WHEN company.invested_companies < 20 THEN 'low_activity'\
    END AS activity\
    FROM fund\
    JOIN company ON company.id=fund.object_id"
    ).na.drop().show()
```

    +--------------------+---------------+
    |                name|       activity|
    +--------------------+---------------+
    |         Brazil Fund|   low_activity|
    |              Fund I|   low_activity|
    |       LionBird Fund|   low_activity|
    |               EVF I|   low_activity|
    |         Second Fund|middle_activity|
    |          First Fund|middle_activity|
    |Early-Stage Fund ...|middle_activity|
    |          Third Fund|middle_activity|
    |Georgetown Alumni...|   low_activity|
    |Georgetown Alumni...|   low_activity|
    |          Formula VC|   low_activity|
    |            Fund III|middle_activity|
    |       Huron Fund IV|   low_activity|
    |Lightspeed China ...|   low_activity|
    |Emerald Cleantech...|   low_activity|
    |Aristos Venture F...|   low_activity|
    |     Innovation Nest|   low_activity|
    |            DWHP III|   low_activity|
    |Private Capital F...|middle_activity|
    |Vine Street Ventu...|   low_activity|
    +--------------------+---------------+
    only showing top 20 rows
    
    

### 9. Среднее количество инвестиционных раундов, в которых фонд принимал участие (выведены категории и среднее число инвестиционных раундов, отсортированные по возрастанию среднего).


```python
spark.sql(
    "SELECT CASE\
    WHEN company.invested_companies >= 100 THEN 'high_activity'\
    WHEN company.invested_companies  >= 20 THEN 'middle_activity'\
    ELSE 'low_activity'\
    END AS activity,\
    ROUND(AVG(company.investment_rounds))\
    FROM fund\
    JOIN company ON company.id=fund.object_id\
    GROUP BY activity\
    ORDER BY ROUND(AVG(company.investment_rounds))"
    ).na.drop().show()
```

    +---------------+--------------------------------+
    |       activity|round(avg(investment_rounds), 0)|
    +---------------+--------------------------------+
    |   low_activity|                             4.0|
    |middle_activity|                            60.0|
    |  high_activity|                           241.0|
    +---------------+--------------------------------+
    
    

### 10. Таблица с десятью самыми активными инвестирующими странами (для каждой страны посчитаны минимальное, максимальное и среднее число компаний, в которые инвестировали фонды, основанные с 2010 по 2012 год включительно; исключены страны с фондами, у которых минимальное число компаний, получивших инвестиции, равно нулю).


```python
spark.sql(
    "SELECT company.country_code,\
    MIN(company.invested_companies),\
    MAX(company.invested_companies),\
    AVG(company.invested_companies) AS avg_invested_companies\
    FROM fund\
    JOIN company ON company.id=fund.object_id\
    WHERE EXTRACT(year FROM CAST(company.founded_at AS date)) BETWEEN '2010' AND '2012'\
    GROUP BY country_code\
    HAVING MIN(invested_companies)>0\
    ORDER BY avg_invested_companies DESC, country_code\
    LIMIT 10"
    ).show()
```

    +------------+-----------------------+-----------------------+----------------------+
    |country_code|min(invested_companies)|max(invested_companies)|avg_invested_companies|
    +------------+-----------------------+-----------------------+----------------------+
    |         CAN|                     29|                     29|                  29.0|
    |         CHL|                     29|                     29|                  29.0|
    |         RUS|                      3|                      9|                   7.0|
    |         IRL|                      5|                      5|                   5.0|
    |         ISR|                      4|                      4|                   4.0|
    |         POL|                      4|                      4|                   4.0|
    |         LBN|                      3|                      3|                   3.0|
    |         HKG|                      2|                      3|                   2.4|
    |         TUR|                      2|                      2|                   2.0|
    +------------+-----------------------+-----------------------+----------------------+
    
    

### 11. Имя и фамилию всех сотрудников компаний (добавлено поле с названием учебного заведения, которое окончил сотрудник, если эта информация известна).


```python
spark.sql(
    "SELECT people.first_name,\
    people.last_name,\
    education.institution\
    FROM people\
    JOIN company ON company.id=people.object_id\
    LEFT JOIN education ON education.object_id=people.object_id"
    ).show()
```

    +----------+-------------+--------------------+
    |first_name|    last_name|         institution|
    +----------+-------------+--------------------+
    |      Mark|   Zuckerberg|  Harvard University|
    |     Peter|       Lester|                null|
    |Dr. Steven|  E. Saunders|                null|
    |      Neil|        Capel|                null|
    |       Sue|       Pilsch|                null|
    |     Keith|Kurzendoerfer|                null|
    |  Courtney|        Homer|MIT Sloan School ...|
    |      Eddy|      Badrina|                null|
    |   Michael|    Dadashyan|                null|
    |      Jeff|        Grell|                null|
    |      Nick|         Bova|                null|
    |     Umesh|        Singh|University of Mumbai|
    |     Umesh|        Singh|  Rutgers University|
    |     Larry|  Blankenship|                null|
    |    Steven|  G. Anderson|                null|
    |    Thomas|  F. Ackerman|                null|
    |      Kurt|   Azarbarzin|                null|
    |      Adam|    Beckerman|University of Mar...|
    |      Adam|    Beckerman|Columbia Universi...|
    |   Melissa|       French|                null|
    +----------+-------------+--------------------+
    only showing top 20 rows
    
    

### 12. Топ-5 компаний по количеству учебных заведений, которые окончили их сотрудники (название компании и число уникальных названий учебных заведений).


```python
spark.sql(
    "WITH tab AS (SELECT pe.object_id,\
    ed.institution\
    FROM people AS pe\
    LEFT JOIN education AS ed ON pe.object_id=ed.object_id)\
    \
    SELECT co.name,\
    COUNT(DISTINCT ta.institution)\
    FROM tab AS ta\
    RIGHT JOIN company AS co ON co.id=ta.object_id\
    GROUP BY co.name\
    ORDER BY COUNT(DISTINCT ta.institution) DESC\
    LIMIT 5"
    ).show()
```

    +-------------+---------------------------+
    |         name|count(DISTINCT institution)|
    +-------------+---------------------------+
    |   James King|                         12|
    |Gaurav Sharma|                         11|
    |  Scott Smith|                         10|
    |  Sean Murphy|                         10|
    | Michael Yang|                         10|
    +-------------+---------------------------+
    
    

### 13. Список уникальных имен персон, аффилированных в компаниях Facebook, Plaxo, YouTube.


```python
spark.sql(
    "WITH\
    tab_c AS (SELECT *\
    FROM company\
    WHERE entity_type = 'Person')\
    \
    SELECT entity_type,\
    name\
    FROM people\
    JOIN tab_c ON people.object_id=tab_c.id\
    WHERE affiliation_name = 'Facebook'\
    OR affiliation_name = 'Plaxo'\
    OR affiliation_name = 'YouTube'"
    ).show()
```

    +-----------+--------------------+
    |entity_type|                name|
    +-----------+--------------------+
    |     Person|     Mark Zuckerberg|
    |     Person|        Peter Lester|
    |     Person|        Mike Cannady|
    |     Person|     J. Todd Masonis|
    |     Person|         John McCrea|
    |     Person|     Cameron T. Ring|
    |     Person|        Joseph Smarr|
    |     Person|       Ruchi Sanghvi|
    |     Person|Venkates Swaminathan|
    |     Person|       Adam Marchick|
    |     Person|           Jimmy Zhu|
    |     Person|    Dustin Moskovitz|
    |     Person|         Shashi Seth|
    |     Person|      Brandon Sayles|
    |     Person|    Alison Rosenthal|
    |     Person|      Jordan Hoffner|
    |     Person|     Makinde Adeagbo|
    |     Person|      Michael Ortali|
    |     Person|     Katherine Losse|
    |     Person|        Emily Grewal|
    +-----------+--------------------+
    only showing top 20 rows
    
    

### 14. Список уникальных номеров сотрудников, которые работают в компаниях, отобранных в предыдущем запросе.


```python
spark.sql(
    "WITH\
    tab_c AS (SELECT *\
    FROM company\
    WHERE entity_type = 'Person')\
    \
    SELECT people.id\
    FROM people\
    JOIN tab_c ON people.object_id=tab_c.id\
    WHERE affiliation_name = 'Facebook'\
    OR affiliation_name = 'Plaxo'\
    OR affiliation_name = 'YouTube'"
    ).show()
```

    +------+
    |    id|
    +------+
    |     9|
    |    87|
    | 87073|
    |    88|
    |    89|
    |    90|
    |    91|
    |  8846|
    |    92|
    | 91314|
    | 92926|
    |    10|
    | 10369|
    |104717|
    |106061|
    | 11219|
    |111414|
    |114680|
    |119060|
    |121604|
    +------+
    only showing top 20 rows
    
    

### 15. Уникальные пары с номерами сотрудников из предыдущего запроса и учебными заведениями, которое окончил сотрудник.


```python
spark.sql(
    "WITH\
    tab_c AS (SELECT *\
    FROM company\
    WHERE entity_type = 'Person')\
    \
    SELECT people.id,\
    education.institution\
    FROM people\
    JOIN tab_c ON people.object_id=tab_c.id\
    JOIN education ON people.object_id=education.object_id\
    WHERE affiliation_name = 'Facebook'\
    OR affiliation_name = 'Plaxo'\
    OR affiliation_name = 'YouTube'"
    ).show()
```

    +------+--------------------+
    |    id|         institution|
    +------+--------------------+
    |     9|  Harvard University|
    | 87073|University of Vir...|
    | 87073|New York Universi...|
    |    89|Massachusetts Ins...|
    |    89| Stanford University|
    |    91| Stanford University|
    |    91| Stanford University|
    |  8846|Carnegie Mellon U...|
    |  8846|Carnegie Mellon U...|
    | 91314| Stanford University|
    | 92926|Massachusetts Ins...|
    |    10|  Harvard University|
    | 10369| University of Poona|
    | 10369|University of Kanpur|
    |104717|National Institut...|
    |106061|    Brown University|
    |106061|Stanford Universi...|
    | 11219|New York Universi...|
    | 11219|      Vassar College|
    |111414|Massachusetts Ins...|
    +------+--------------------+
    only showing top 20 rows
    
    

### 16. Количество учебных заведений для каждого сотрудника из предыдущего запроса.


```python
spark.sql(
    "WITH\
    tab_c AS (SELECT *\
    FROM company\
    WHERE entity_type = 'Person')\
    \
    SELECT people.id,\
    COUNT(education.institution)\
    FROM people\
    JOIN tab_c ON people.object_id=tab_c.id\
    JOIN education ON people.object_id=education.object_id\
    WHERE affiliation_name = 'Facebook'\
    OR affiliation_name = 'Plaxo'\
    OR affiliation_name = 'YouTube'\
    GROUP BY people.id"
    ).show()
```

    +------+------------------+
    |    id|count(institution)|
    +------+------------------+
    |   148|                 1|
    |114680|                 1|
    | 24482|                 1|
    | 87073|                 2|
    | 56219|                 1|
    | 91314|                 1|
    | 58345|                 1|
    |150686|                 2|
    | 37481|                 3|
    |    91|                 2|
    |  8846|                 2|
    |147841|                 2|
    | 52352|                 1|
    |134176|                 1|
    |111414|                 1|
    | 52351|                 2|
    |164843|                 1|
    |104717|                 1|
    | 36520|                 1|
    | 39752|                 1|
    +------+------------------+
    only showing top 20 rows
    
    

### 17. Среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники разных компаний.


```python
spark.sql(
    "WITH\
    tab_c AS (SELECT *\
    FROM company\
    WHERE entity_type = 'Person'),\
    \
    tab_ct AS (SELECT people.id,\
    COUNT(education.institution) AS coun\
    FROM people\
    JOIN tab_c ON people.object_id=tab_c.id\
    JOIN education ON people.object_id=education.object_id\
    WHERE affiliation_name = 'Facebook'\
    OR affiliation_name = 'Plaxo'\
    OR affiliation_name = 'YouTube'\
    GROUP BY people.id)\
    \
    SELECT AVG(tab_ct.coun)\
    FROM tab_ct"
    ).show()
```

    +-----------------+
    |        avg(coun)|
    +-----------------+
    |1.540983606557377|
    +-----------------+
    
    

### 18. Среднее число учебных заведений (всех, не только уникальных), которые окончили только сотрудники Facebook.


```python
spark.sql(
    "WITH\
    tab_c AS (SELECT *\
    FROM company\
    WHERE entity_type = 'Person'),\
    \
    tab_ct AS (SELECT people.id,\
    COUNT(education.institution) AS coun\
    FROM people\
    JOIN tab_c ON people.object_id=tab_c.id\
    JOIN education ON people.object_id=education.object_id\
    WHERE affiliation_name = 'Facebook'\
    GROUP BY people.id)\
    \
    SELECT AVG(tab_ct.coun)\
    FROM tab_ct"
    ).show()
```

    +-----------------+
    |        avg(coun)|
    +-----------------+
    |1.511111111111111|
    +-----------------+
    
    

### 19. Таблица с полями:
* name_of_fund — название фонда;
* name_of_company — название компании;
* amount — сумма инвестиций, которую привлекла компания в раунде.

В таблице данные о компаниях, в истории которых было больше шести важных этапов, а раунды финансирования проходили с 2012 по 2013 год включительно.


```python
spark.sql(
    "SELECT DISTINCT fund.name AS name_of_fund,\
    company.name AS name_of_company,\
    funding_round.raised_amount AS amount\
    FROM investment\
    INNER JOIN fund ON fund.object_id=investment.investor_object_id\
    INNER JOIN funding_round ON funding_round.funding_round_id=investment.funding_round_id\
    INNER JOIN company ON company.id=investment.investor_object_id\
    WHERE company.id IN (SELECT id\
    FROM company\
    WHERE milestones > 6)\
    AND funding_round.object_id IN (SELECT object_id\
    FROM funding_round\
    WHERE EXTRACT(YEAR FROM CAST(funded_at AS DATE)) BETWEEN 2012 AND 2013)"
    ).show()
```

    +--------------------+--------------------+-----------+
    |        name_of_fund|     name_of_company|     amount|
    +--------------------+--------------------+-----------+
    |Bessemer Venture ...|Bessemer Venture ...|  7500000.0|
    |   MPM BioVentures V|         MPM Capital|      3.0E7|
    |Bessemer Venture ...|Bessemer Venture ...|  6360000.0|
    |$550M HealthCare ...|         MPM Capital|     4.04E7|
    |Bessemer Venture ...|Bessemer Venture ...|  1200000.0|
    |Bessemer Venture ...|Bessemer Venture ...|      2.0E7|
    |Bessemer Venture ...|Bessemer Venture ...|     1.82E7|
    |Alsop Louie Partn...|Alsop Louie Partners|      2.0E7|
    |Bessemer Venture ...|Bessemer Venture ...|      2.6E7|
    |Alsop Louie Capit...|Alsop Louie Partners|      2.0E7|
    |   MPM BioVentures V|         MPM Capital|      6.6E7|
    |   MPM BioVentures V|         MPM Capital|5.7949892E7|
    |Bessemer Venture ...|Bessemer Venture ...|  8000000.0|
    |Bessemer Venture ...|Bessemer Venture ...|     2.95E7|
    |$550M HealthCare ...|         MPM Capital|  5000000.0|
    |$550M HealthCare ...|         MPM Capital|      2.8E7|
    |Alsop Louie Partn...|Alsop Louie Partners|  3000000.0|
    |Bessemer Venture ...|Bessemer Venture ...|      3.8E7|
    |Alsop Louie Capit...|Alsop Louie Partners|  1000000.0|
    |Bessemer Venture ...|Bessemer Venture ...|     1.15E7|
    +--------------------+--------------------+-----------+
    only showing top 20 rows
    
    

### 20. Таблица, с полями:
* название компании-покупателя;
* сумма сделки;
* название компании, которую купили;
* сумма инвестиций, вложенных в купленную компанию;
* доля, которая отображает, во сколько раз сумма покупки превысила сумму вложенных в компанию инвестиций, округлённая до ближайшего целого числа.

Не учтены те сделки, в которых сумма покупки равна нулю. Если сумма инвестиций в компанию равна нулю, такие компании из таблицы исключены. Таблица отсортирована по сумме сделки от большей к меньшей, а затем по названию купленной компании в лексикографическом порядке. Таблица ограничена первыми десятью записями.


```python
spark.sql(
    "SELECT acquiring.name AS acquiring_company_name,\
    acquisition.price_amount AS amount,\
    acquired.name AS acquired_company_name,\
    acquired.funding_total_usd AS f_total,\
    ROUND(acquisition.price_amount/acquired.funding_total_usd)\
    FROM acquisition\
    LEFT OUTER JOIN company AS acquired ON acquisition.acquired_object_id = acquired.id\
    LEFT OUTER JOIN company AS acquiring ON acquisition.acquiring_object_id = acquiring.id\
    WHERE price_amount != 0\
    AND acquired.funding_total_usd !=0\
    ORDER BY amount DESC, acquired_company_name\
    LIMIT 10"
    ).show()
```

    +----------------------+------+---------------------+-----------+--------------------------------------------+
    |acquiring_company_name|amount|acquired_company_name|    f_total|round((price_amount / funding_total_usd), 0)|
    +----------------------+------+---------------------+-----------+--------------------------------------------+
    |            Scout Labs| 4.9E9| Varian Semiconduc...|  4800000.0|                                      1021.0|
    |              Broadcom| 3.7E9|              Aeluros|  7970000.0|                                       464.0|
    |              Broadcom| 3.7E9| NetLogic Microsys...|188527015.0|                                        20.0|
    |  Level 3 Communica...| 3.0E9|      Global Crossing| 41000000.0|                                        73.0|
    |            Salesforce| 2.5E9|          ExactTarget|238209999.0|                                        10.0|
    |     Johnson & Johnson| 2.3E9|              Crucell|443000000.0|                                         5.0|
    |     Micron Technology|1.27E9|              Numonyx|150000000.0|                                         8.0|
    |              Eutelsat|1.14E9|               Satmex| 96250000.0|                                        12.0|
    |                Google| 1.1E9|                 Waze| 67000000.0|                                        16.0|
    |     Johnson & Johnson| 1.0E9| Aragon Pharmaceut...|122000000.0|                                         8.0|
    +----------------------+------+---------------------+-----------+--------------------------------------------+
    
    

### 21. Таблица с названиями компаний из категории social, получившие финансирование с 2010 по 2013 год включительно. Выведен также номер месяца, в котором проходил раунд финансирования (сумма инвестиций не равна нулю).


```python
spark.sql(
    "SELECT c.name,\
    EXTRACT(MONTH FROM CAST(fr.funded_at AS date)) AS month\
    FROM company AS c\
    JOIN funding_round AS fr ON c.id = fr.object_id\
    WHERE c.category_code LIKE '%social%'\
    AND EXTRACT(YEAR FROM CAST(fr.funded_at AS date)) BETWEEN '2010' AND '2013'\
    AND fr.raised_amount_usd > 0"
    ).show()
```

    +---------------+-----+
    |           name|month|
    +---------------+-----+
    |         gopogo|    4|
    |      CrushBlvd|    1|
    |Collective Bias|    4|
    |       RuffWire|    6|
    |        Get.com|    9|
    |   Academia.edu|    9|
    |   Academia.edu|   11|
    |   Academia.edu|    4|
    |      ev-social|    7|
    |        Twitter|    9|
    |        Twitter|    8|
    |        Twitter|   12|
    |        Twitter|    1|
    |     GrupHediye|    1|
    |   ResearchGate|    6|
    |        Ondango|    9|
    |        Ondango|    8|
    |       SocialGO|    2|
    |       SocialGO|    1|
    |          QWiPS|   10|
    +---------------+-----+
    only showing top 20 rows
    
    

### 22. Данные по месяцам с 2010 по 2013 год, когда проходили инвестиционные раунды. Данные сгруппированы по номеру месяца с полями:
* номер месяца, в котором проходили раунды;
* количество уникальных названий фондов из США, которые инвестировали в этом месяце;
* количество компаний, купленных за этот месяц;
* общая сумма сделок по покупкам в этом месяце.


```python
spark.sql(
    "WITH\
    s1 AS (SELECT EXTRACT(MONTH FROM CAST(fr.funded_at AS date)) AS month_f,\
    COUNT(DISTINCT f.name) AS count_name\
    FROM fund as f\
    INNER JOIN investment AS i ON f.object_id = i.investor_object_id\
    INNER JOIN funding_round AS fr ON fr.funding_round_id = i.funding_round_id\
    INNER JOIN company AS co ON co.id = i.investor_object_id\
    WHERE EXTRACT(YEAR FROM CAST(fr.funded_at AS date)) BETWEEN '2010' AND '2013'\
    AND co.country_code = 'USA'\
    GROUP BY month_f),\
    \
    s2 AS (SELECT EXTRACT(MONTH FROM CAST(acquired_at AS date)) AS month_as,\
    COUNT(acquired_object_id) AS count_ac,\
    SUM(price_amount) AS sum_pr\
    FROM acquisition\
    WHERE EXTRACT(YEAR FROM CAST(acquired_at AS date)) BETWEEN '2010' AND '2013'\
    GROUP BY month_as)\
    \
    SELECT s2.month_as,\
    count_name,\
    count_ac,\
    sum_pr\
    FROM s1\
    JOIN s2 ON s1.month_f = s2.month_as"
    ).show()
```

    +--------+----------+--------+---------------+
    |month_as|count_name|count_ac|         sum_pr|
    +--------+----------+--------+---------------+
    |      12|       132|     437| 4.033359415E10|
    |       1|       167|     605|2.7120983206E10|
    |       6|       153|     542| 5.446979015E10|
    |       3|       140|     467| 5.969051267E10|
    |       5|       149|     543|    8.688131E10|
    |       9|       155|     504|7.0653865061E10|
    |       4|       154|     419|3.0601314111E10|
    |       8|       157|     462|7.8004015001E10|
    |       7|       156|     497|5.0084584358E10|
    |      10|       155|     480|5.0246690433E10|
    |      11|       151|     422|4.8519311202E10|
    |       2|       142|     422|  4.15299639E10|
    +--------+----------+--------+---------------+
    
    

### 23. Сводная таблица со средней суммой инвестиций для стран, в которых есть стартапы, зарегистрированные в 2011, 2012 и 2013 годах. Данные за каждый год в отдельном поле. Таблица отсортирована по среднему значению инвестиций за 2011 год от большего к меньшему.


```python
spark.sql(
    "WITH\
    first AS (SELECT country_code,\
    ROUND(AVG(funding_total_usd), 2) avg_ft_2011,\
    '2011' AS year\
    FROM company\
    WHERE id IN (SELECT id FROM company WHERE EXTRACT(year FROM founded_at) = 2011)\
    GROUP BY country_code),\
    \
    second AS (SELECT country_code,\
    ROUND(AVG(funding_total_usd), 2) avg_ft_2012,\
    '2012' AS year\
    FROM company\
    WHERE id IN (SELECT id FROM company WHERE EXTRACT(year FROM founded_at) = 2012)\
    GROUP BY country_code),\
    \
    third AS (SELECT country_code,\
    ROUND(AVG(funding_total_usd), 2) avg_ft_2013,\
    '2013' AS year\
    FROM company\
    WHERE id IN (SELECT id FROM company WHERE EXTRACT(year FROM founded_at) = 2013)\
    GROUP BY country_code)\
    \
    SELECT f.country_code,\
    f.avg_ft_2011,\
    s.avg_ft_2012,\
    t.avg_ft_2013\
    FROM first AS f\
    JOIN second AS s ON f.country_code = s.country_code\
    JOIN third AS t ON f.country_code = t.country_code\
    ORDER BY avg_ft_2011  DESC"
    ).show()
```

    +------------+-----------+-----------+-----------+
    |country_code|avg_ft_2011|avg_ft_2012|avg_ft_2013|
    +------------+-----------+-----------+-----------+
    |         USA| 2805455.51| 1287942.43| 1272036.14|
    |         AUS| 2497378.56|   261400.0|   44826.41|
    |         FRA| 2049447.65|   61445.17|  1515799.6|
    |         CHN|  2000000.0|   54746.15|  2500000.0|
    |         ARG|  1661000.0|   275000.0|        0.0|
    |         GBR| 1569421.71|  416066.22|   521656.2|
    |         KOR|  1000000.0|   289000.0|        0.0|
    |         JPN|  998141.06|  827529.94|   145000.0|
    |         RUS|  910092.59|   536500.0|   14444.44|
    |         ANT|   775000.0|        0.0|    19299.0|
    |         IND|  737295.08|  351848.79|   64677.42|
    |         SGP|  515454.55|  611047.62|  153172.73|
    |         BRA|  487692.33|   49822.82|   50952.38|
    |         BEL|   472219.0|        0.0|        0.0|
    |         ISR|  433333.33| 1342647.06|    10000.0|
    |         GRC|   402500.0|    71340.5|        0.0|
    |         DEU|  399355.11|  123794.24|   27343.75|
    |         THA|   343750.0|        0.0|   100000.0|
    |         ARE|   340000.0|   600000.0|        0.0|
    |         PRT|   328119.0|        0.0|        0.0|
    +------------+-----------+-----------+-----------+
    only showing top 20 rows
    
    

## Результаты исследования

**Выполнены SQL – запросы, в результате которых сформированы таблицы со следующей информацией:**

1. Количество компаний, которые закрылись.
2. Количество привлечённых средств для новостных компаний США.
3. Общая сумма сделок по покупке одних компаний другими (сделки, которые осуществлялись только за наличные с 2011 по 2013 год включительно).
4. Имя, фамилия и название аккаунтов людей в твиттере, у которых названия аккаунтов начинаются на 'Silver'.
5. Информация о людях, у которых названия аккаунтов в твиттере содержат подстроку 'money', а фамилия начинается на 'K'.
6. Общая сумма привлечённых инвестиций, которые получили компании, зарегистрированные в стране.
7. Дата проведения раунда, а также минимальное и максимальное значения суммы инвестиций, привлечённых в эту дату (в итоговой таблице только те записи, в которых минимальное значение суммы инвестиций не равно нулю и не равно максимальному значению).
8. Поле с категориями:
  *	для фондов, которые инвестируют в 100 и более компаний - high_activity;
  *	для фондов, которые инвестируют в 20 и более компаний до 100 - middle_activity;
  *	если количество инвестируемых компаний фонда не достигает 20 - low_activity.
  
  (отображены поле name таблицы fund и поле с категориями)
  
9. Среднее количество инвестиционных раундов, в которых фонд принимал участие (выведены категории и среднее число инвестиционных раундов, отсортированные по возрастанию среднего).
10. Таблица с десятью самыми активными инвестирующими странами (для каждой страны посчитаны минимальное, максимальное и среднее число компаний, в которые инвестировали фонды, основанные с 2010 по 2012 год включительно; исключены страны с фондами, у которых минимальное число компаний, получивших инвестиции, равно нулю).
11. Имя и фамилию всех сотрудников компаний (добавлено поле с названием учебного заведения, которое окончил сотрудник, если эта информация известна).
12. Топ-5 компаний по количеству учебных заведений, которые окончили их сотрудники (название компании и число уникальных названий учебных заведений).
13. Список уникальных имен персон, аффилированных в компаниях Facebook, Plaxo, YouTube.
14. Список уникальных номеров сотрудников, которые работают в компаниях, отобранных в предыдущем запросе.
15. Уникальные пары с номерами сотрудников из предыдущего запроса и учебными заведениями, которое окончил сотрудник.
16. Количество учебных заведений для каждого сотрудника из предыдущего запроса.
17. Среднее число учебных заведений (всех, не только уникальных), которые окончили сотрудники разных компаний.
18. Среднее число учебных заведений (всех, не только уникальных), которые окончили только сотрудники Facebook.
19. Таблица с полями:
  *	name_of_fund — название фонда;
  *	name_of_company — название компании;
  *	amount — сумма инвестиций, которую привлекла компания в раунде.
  
  В таблице данные о компаниях, в истории которых было больше шести важных этапов, а раунды финансирования проходили с 2012 по 2013 год включительно.
  
20. Таблица, с полями:
  *	название компании-покупателя;
  *	сумма сделки;
  *	название компании, которую купили;
  *	сумма инвестиций, вложенных в купленную компанию;
  *	доля, которая отображает, во сколько раз сумма покупки превысила сумму вложенных в компанию инвестиций, округлённая до ближайшего целого числа.
  
  Не учтены те сделки, в которых сумма покупки равна нулю. Если сумма инвестиций в компанию равна нулю, такие компании из таблицы исключены. Таблица отсортирована по   сумме сделки от большей к меньшей, а затем по названию купленной компании в лексикографическом порядке. Таблица ограничена первыми десятью записями.
  
21. Таблица с названиями компаний из категории social, получившие финансирование с 2010 по 2013 год включительно. Выведен также номер месяца, в котором проходил раунд финансирования (сумма инвестиций не равна нулю).
22. Данные по месяцам с 2010 по 2013 год, когда проходили инвестиционные раунды. Данные сгруппированы по номеру месяца с полями:
  *	номер месяца, в котором проходили раунды;
  *	количество уникальных названий фондов из США, которые инвестировали в этом месяце;
  *	количество компаний, купленных за этот месяц;
  *	общая сумма сделок по покупкам в этом месяце.
23. Сводная таблица со средней суммой инвестиций для стран, в которых есть стартапы, зарегистрированные в 2011, 2012 и 2013 годах. Данные за каждый год в отдельном поле. Таблица отсортирована по среднему значению инвестиций за 2011 год от большего к меньшему.
