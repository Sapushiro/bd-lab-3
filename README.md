# Banknote Authentication Service with HashiCorp Vault

Проект продолжает ML-сервис из второй лабораторной и добавляет
централизованное хранение конфигурации подключения к Microsoft SQL
Server в HashiCorp Vaultю

Сервис принимает числовые характеристики банкноты, выполняет бинарную
классификацию с помощью обученной модели Logistic Regression, сохраняет
результат в MS SQL Server, а параметры подключения к базе получает из
HashiCorp Vault.

## Датасет

Используется датасет [Bank Note Authentication UCI
Data](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data).

Датасет содержит 1372 наблюдения. Признаки получены в результате
вейвлет-преобразования изображений банкнот. Целевая переменная
определяет подлинность банкноты.

| Поле | Описание |
|---|---|
| `variance` | Дисперсия преобразованного изображения |
| `skewness` | Асимметрия преобразованного изображения |
| `curtosis` | Эксцесс преобразованного изображения |
| `entropy` | Энтропия изображения |
| `class` | Целевой класс: `0` — подлинная, `1` — поддельная |

Во время предобработки удаляются дубликаты. После очистки остаётся 1348
объектов. Данные разделяются на обучающую и тестовую выборки в отношении
70/30 с сохранением соотношения классов (`stratify=y`). Для
воспроизводимости используется `random_state=0`.

## Архитектура решения

В Docker Compose используются четыре сервиса:

| Сервис | Назначение | Порт |
|---|---|---|
| `api` | FastAPI, ML-модель и слой доступа к данным | `8000` |
| `mssql` | Microsoft SQL Server 2022 | `1433` |
| `vault` | HashiCorp Vault | `8200` |
| `vault-init` | Одноразовая инициализация Vault | — |

Docker Compose создаёт общую сеть. Внутри неё сервисы доступны по именам
`mssql` и `vault`.

## Хранение секретов в Vault

Конфигурация подключения к MS SQL Server хранится в KV-хранилище Vault
по пути:

``` text
secret/database
```

В секрете находятся параметры:

``` text
host
port
database
username
password
```

API больше не получает `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` и
`DB_PASSWORD` напрямую через `.env`. При инициализации класса `Database`
приложение подключается к Vault с помощью библиотеки `hvac`, читает
`secret/database` и на основе полученных значений создаёт SQLAlchemy
Engine.

Цепочка подключения:

``` text
FastAPI
   ↓
HashiCorp Vault
   ↓
secret/database
   ↓
SQLAlchemy → pyodbc → Microsoft ODBC Driver 18
   ↓
MS SQL Server
```

## Инициализация Vault

Vault запускается в dev-режиме. Для автоматической подготовки хранилища
используется отдельный одноразовый контейнер `vault-init`.

После запуска `vault-init.sh`:

1.  ожидает готовности Vault;
2.  записывает конфигурацию БД в `secret/database`;
3.  создаёт policy `banknote-api`;
4.  создаёт ограниченный token для API;
5.  записывает API token в общий Docker volume `vault-auth-data`;
6.  завершает работу.

Скрипт использует `set -e`, поэтому при ошибке инициализация завершается
с ненулевым exit code. API зависит от `vault-init` через
`condition: service_completed_successfully` и запускается только после
успешной инициализации Vault.

### Политика доступа API

API не использует root token Vault. Для него создаётся отдельная policy,
разрешающая чтение конфигурации БД из `secret/database`.

Таким образом:

``` text
API token
├── READ secret/database  → разрешено
└── WRITE secret/database → запрещено
```

Token передаётся API не через переменную `VAULT_TOKEN`, а через файл:

``` text
/vault-auth/api-token
```

Путь к нему задаётся переменной:

``` text
VAULT_TOKEN_FILE=/vault-auth/api-token
```

Для передачи файла между `vault-init` и `api` используется именованный
Docker volume `vault-auth-data`. В API он подключается в режиме
`read-only`.

## Bootstrap-секреты

Для первоначального запуска Vault и MS SQL Server необходимы
bootstrap-секреты:

``` text
VAULT_ROOT_TOKEN
MSSQL_SA_PASSWORD
```

Локально они находятся в файле:

``` text
vault-bootstrap.env
```

Файл добавлен в `.gitignore` и не хранится в репозитории.

Bootstrap-секреты используются только для первоначальной настройки
инфраструктуры. API не получает `VAULT_ROOT_TOKEN` или
`MSSQL_SA_PASSWORD` напрямую.

В GitHub Actions значения берутся из GitHub Secrets и временно
записываются в `vault-bootstrap.env` на runner.

## Структура проекта

``` text
bd-lab-3/
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── .dvc/
├── data/
├── experiments/
├── notebooks/
│   └── BankNote_classification.ipynb
├── src/
│   ├── unit_tests/
│   ├── api.py
│   ├── database.py
│   ├── logger.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── tests/
├── .dvcignore
├── .gitignore
├── config.ini
├── data.dvc
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── scenario.json
├── vault-bootstrap.env        # локально, не хранится в Git
└── vault-init.sh
```

Файлы датасета и обученной модели версионируются с помощью DVC.
Bootstrap-файл с секретами исключён из Git.

## Настройка и запуск

### 1. Клонирование репозитория

``` bash
git clone https://github.com/Sapushiro/bd-lab-3.git
cd bd-lab-3
```

### 2. Получение данных и модели

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
dvc pull -r origin
```

Для локального доступа к DagsHub используется конфигурация DVC в
`.dvc/config.local`, которая не хранится в Git.

### 3. Создание bootstrap-конфигурации

Создайте локальный файл `vault-bootstrap.env`:

``` env
VAULT_ROOT_TOKEN=change_me
MSSQL_SA_PASSWORD=change_me
```

Файл не должен добавляться в Git.

### 4. Запуск Docker Compose

``` bash
docker compose --env-file vault-bootstrap.env up -d --build
```

Проверка состояния:

``` bash
docker compose ps -a
```

Ожидаемое состояние:

``` text
api         Up / healthy
mssql       Up / healthy
vault       Up
vault-init  Exited (0)
```

`Exited (0)` для `vault-init` является нормальным состоянием: контейнер
выполняет одноразовую инициализацию и завершает работу.

Логи инициализации:

``` bash
docker logs banknote-vault-init-container
```

Swagger UI доступен по адресу `http://localhost:8000/docs`, Vault UI ---
по адресу `http://localhost:8200`.

Остановка:

``` bash
docker compose --env-file vault-bootstrap.env down
```

Остановка с удалением volumes:

``` bash
docker compose --env-file vault-bootstrap.env down -v
```

## REST API

### `GET /health`

Проверяет доступность API.

``` json
{
  "status": "ok"
}
```

### `POST /predict`

Выполняет классификацию и сохраняет результат в MS SQL Server.

``` bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "variance": 3.6216,
    "skewness": 8.6661,
    "curtosis": -2.8073,
    "entropy": -0.44699
  }'
```

Пример ответа:

``` json
{
  "prediction": 0,
  "label": "authentic"
}
```

### `GET /predictions`

Возвращает сохранённую историю предсказаний:

``` bash
curl http://localhost:8000/predictions
```

## Подготовка данных и обучение

Предобработка:

``` bash
python -m src.preprocess
```

Обучение:

``` bash
python -m src.train
```

Обученная модель сохраняется в `experiments/log_reg.sav`.

Для классификации используется `LogisticRegression` из scikit-learn.

  Метрика                     Значение
  --------------------- --------------
  Accuracy                    `0.9975`
  Верных предсказаний     `404 из 405`

Матрица ошибок:

``` text
[[216   1]
 [  0 188]]
```

## Тестирование

### Модульные тесты

Модульные тесты запускаются внутри Docker-образа API:

``` bash
python -m unittest discover -s src/unit_tests -v
```

Внешние зависимости подменяются в unit-тестах, поэтому для их выполнения
не требуется запуск настоящих Vault и MS SQL Server.

### Функциональные тесты

Функциональные тесты выполняются после запуска всей инфраструктуры:

``` text
Vault + vault-init + MS SQL Server + API
```

Они проверяют работу REST API и взаимодействие с настоящей базой данных.
Таким образом проверяется полная цепочка:

``` text
POST /predict
      ↓
API читает конфигурацию из Vault
      ↓
ML-модель
      ↓
INSERT в MS SQL Server
      ↓
GET /predictions
      ↓
проверка сохранённой записи
```

В CD результаты записываются в `functional-test-results.txt` и
публикуются как артефакт GitHub Actions.

## Docker

API собирается из `Dockerfile` на основе Python-образа. В контейнер
устанавливаются Microsoft ODBC Driver 18, `pyodbc`, SQLAlchemy, `hvac`,
зависимости Python, исходный код и обученная модель.

Используемые внешние образы:

``` text
mcr.microsoft.com/mssql/server:2022-latest
hashicorp/vault:latest
```

В `docker-compose.yml` настроены:

-   сервис `api`;
-   сервис `mssql`;
-   сервис `vault`;
-   одноразовый сервис `vault-init`;
-   healthcheck SQL Server;
-   healthcheck API;
-   именованный volume `mssql-data`;
-   именованный volume `vault-auth-data`;
-   запуск API после успешного `vault-init` и готовности MS SQL Server.

Опубликованный образ API:
[sapushiro/banknote-api-lab3](https://hub.docker.com/r/sapushiro/banknote-api-lab3).

## DVC и DagsHub

DVC используется для версионирования директории `data/` и обученной
модели `experiments/log_reg.sav`.

Для локальной авторизации DVC используется `.dvc/config.local`. В GitHub
Actions доступ к DagsHub осуществляется через GitHub Secrets:

``` text
DAGSHUB_USERNAME
DAGSHUB_TOKEN
```

## CI/CD

### Continuous Integration

Workflow `.github/workflows/ci.yml` запускается при создании Pull
Request в `main` или вручную.

Он:

1.  получает исходный код;
2.  устанавливает DVC;
3.  настраивает доступ к DagsHub;
4.  загружает данные и модель;
5.  собирает Docker-образ API;
6.  запускает модульные тесты внутри контейнера;
7.  авторизуется в DockerHub;
8.  публикует `sapushiro/banknote-api-lab3` с тегами `latest` и SHA
    коммита.

Vault и MS SQL Server на этапе CI не запускаются, так как CI выполняет
сборку и изолированные модульные тесты.

### Continuous Delivery

Workflow `.github/workflows/cd.yml` запускается после успешного CI или
вручную.

Он:

1.  получает исходный код;
2.  создаёт временный `vault-bootstrap.env` из GitHub Secrets;
3.  загружает готовый API-образ из DockerHub;
4.  запускает Vault, `vault-init`, MS SQL Server и API через Docker
    Compose;
5.  проверяет успешное завершение `vault-init`;
6.  ожидает успешного healthcheck API;
7.  выполняет функциональные тесты;
8.  сохраняет `functional-test-results.txt` как артефакт;
9.  выводит Docker-логи при ошибке;
10. удаляет тестовые контейнеры и volumes.

Используемые GitHub Secrets:

``` text
DAGSHUB_USERNAME
DAGSHUB_TOKEN
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
MSSQL_SA_PASSWORD
VAULT_ROOT_TOKEN
```

`DAGSHUB_TOKEN` и `DOCKERHUB_TOKEN` являются инфраструктурными секретами
CI/CD и хранятся в GitHub Secrets. Vault используется для конфигурации,
необходимой работающему приложению при подключении к базе данных.

## Ссылки

-   [GitHub-репозиторий](https://github.com/Sapushiro/bd-lab-3)
-   [Датасет на
    Kaggle](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data)
-   [DagsHub-хранилище](https://dagshub.com/Sapushiro/bd-lab-1)
-   [DockerHub-образ](https://hub.docker.com/r/sapushiro/banknote-api-lab3)
