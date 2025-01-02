.venv/bin/sensevoice2jsonl \
++scp_file_list='["data/list/train_wav.scp", "data/list/train_text.txt", "data/list/train_text_language.txt", "data/list/train_emo.txt", "data/list/train_event.txt"]' \
++data_type_list='["source", "target", "text_language", "emo_target", "event_target"]' \
++jsonl_file_out="data/train.jsonl"