# Прогнозирование количества заказов такси на следующий час

## Описание проекта

Компания такси собрала исторические данные о заказах такси в аэропортах. 
Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час.
Строится модель для такого предсказания.

## Ключевые слова проекта

`временные ряды` `регрессия` `подбор гиперпараметров` `выбор модели МО`

## Навыки и инструменты

`исследовательский анализ данных` `Python` `Pandas` `NumPy` `Statsmodels` `Matplotlib` `Scikit-learn` `Lightgbm` `Catboost`

## Результаты исследования

**I. В результате предварительного анализа, установлено:**

1. Данные представлены за период с 2018-03-01 пo 2018-08-31.
2. Данные насчитывают порядка 26 тыс. строк.
3. Данные представлены с разбивкой по 10 мин.
4. Пропуски в данных отсутствуют.

**II. В результате анализа данных, установлено:**

1. Данные имеют общий тренд плавного увеличения от марта к августу.
2. Сезонности в данных присутствует (по дням).
3. Остатки представлены временным рядом близким по своей форме к стохастическому стационарному процессу.

![image](https://user-images.githubusercontent.com/104613549/181461563-8f262f14-b0c3-46ad-8c45-203775b03e8d.png)

**III. В результате обучения, установлено:**

1. Модель: LinearRegression();
RMSE: 25;
CPU times: user 19.3 ms.
2. Лучшие параметры модели: RandomForestRegressor(max_depth=5, n_estimators=5, random_state=12345);
RMSE: 24;
CPU times: user 2 s.
3. Лучшие параметры модели: DecisionTreeRegressor(max_depth=5, random_state=12345);
RMSE: 25;
CPU times: user 1.3 s.
4. Модель: DummyRegressor();
RMSE: 39;
CPU times: user 11.4 ms.
5. Лучшие параметры модели: LGBMRegressor(max_depth=5, n_estimators=5, random_state=12345);
RMSE: 30;
CPU times: user 3.23 s.
6. Модель: CatBoostRegressor();
RMSE: 23;
CPU times: user 2.97 s.

**IV. В результате тестирования, установлено:**

1. Наилучшая модель на ТЕСТОВОЙ выборке: LinearRegression().
2. RMSE наилучшей модели = 48.
 
![image](https://user-images.githubusercontent.com/104613549/181461725-45c269b8-abad-4fc9-ac60-39b968c7e54d.png)
 
 ## Статус проекта
 `Завершен`
