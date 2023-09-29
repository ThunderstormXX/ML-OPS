# Описание

Добавил линейную регрессию, загрузчик данных, обучение и тест модели

Запускать main.py (poetry run python3 main.py)



# ЗАМЕТКИ ПО ВСЕМУ
poetry build 'dir'
poetry init 'dir' (if in exists project)


poetry run python3 'file' (no module named numpy) --> решилось с помощью :
1) poetry shell
2) poetry update

<!-- poetry config virtualenvs.in-project true (переносит виртуальное окружение в директорию) -->

poetry env use  /home/igoreshka/miniconda3/envs/gt/bin/python3.8 (Выбор окружения)

Добавить новый модуль : poetry add 'name' ( не нужен , можно прописать в dependences и сделать poetry update )

Как импортить пакет из другой директории ?

pip install pre-commit
pre-commit install (in first)
pre-commit run --all-files (Не работает)


## Как комитить

git add -A
git commit
pre-commit run --all-files
git push
