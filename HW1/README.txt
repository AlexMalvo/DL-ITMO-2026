ДЗ-1. Свёртки, обучение глубоких сетей, adversarial robustness

Начните с hw1.ipynb: там находятся условия, критерии, открытые тесты и места для ответов.

Запуск
1. Распакуйте весь архив в отдельный каталог.
2. Откройте hw1.ipynb в Jupyter, VS Code или Colab.
3. Установите зависимости в выбранное окружение/kernel:
   python -m pip install -r requirements.txt
   В notebook при необходимости: %pip install -r requirements.txt
   Версии проверены вместе с Python 3.13.
4. Выполняйте notebook по порядку, заполняя код и ответы.
   CIFAR-10 и FashionMNIST загрузятся автоматически в ./data при первом запуске;
   для загрузки нужен интернет. Датасеты не входят в архив.

Выбор задач и сдача
- Основная часть: разделы 1–4, 5.1 и 6; дополнительно выберите одну ветку 5.2/5.3
  через ATTACK_TRACK. Бонусы выбираются через BONUS_TASKS.
- Для отладки оставьте FAST_MODE = True. Финальные эксперименты выполняйте с False.
- Для полной сдачи: Restart & Run All, сохранённый notebook с outputs и каталог artifacts.
  Экспорт artifacts находится перед бонусами. Сохраняйте cell IDs, metadata и сигнатуры.
- При частичной сдаче укажите нерешённые задачи и приложите доступные результаты;
  точные правила приведены в начале notebook.
- Срок выполнения — три недели. Конкретную дату дедлайна сообщает преподаватель.

Для обучения рекомендуется GPU. На CPU adversarial training и серии PGD могут
занимать существенно больше времени. Укажите использованное устройство в notebook.

Запуск решения
- Основной файл: hw1.ipynb. Выбрана targeted attack (5.2).
- Включены бонусы 7.1, 7.2, 7.4 и 7.5: всего до 4 дополнительных баллов.
- Полный прогон из нового kernel с сохранением outputs после каждой ячейки:
  python run_notebook.py
- Отладочный прогон с уменьшенным бюджетом:
  python run_notebook.py --fast
  Его результат записывается в hw1-fast.ipynb. Файлы artifacts при этом также
  перезаписываются; перед сдачей выполните полный прогон ещё раз.
- Итоговые файлы: hw1.ipynb, artifacts/metrics.json, три checkpoint .pt
  и artifacts/additional_metrics.json с результатами бонусов и малых опытов.
- Данные в data/ загружаются автоматически и в Git не включаются.

Окружение GPU-прогона
- Windows, Python 3.13.9, NVIDIA RTX 3090 24 GB, PyTorch 2.11.0+cu130.
- CUDA-сборка для использованного драйвера устанавливается до остальных зависимостей:
  python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
  python -m pip install -r requirements.txt
- Для импорта PyTorch на Windows нужен актуальный Microsoft Visual C++ v14 Runtime:
  https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist
