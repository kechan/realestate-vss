#!/bin/zsh
# ps aux | grep 'celeryd' | awk '{print $2}' | xargs kill -9

celery -A celery_unstack.celery worker --loglevel=info --logfile=celery_unstack.log -Q unstack_queue --detach --hostname=unstack_worker@%h
celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log -P solo -Q embed_queue --detach --hostname=embed_worker@%h 
celery -A celery_update_embeddings.celery worker --loglevel=info --logfile=celery_update_embeddings.log -P solo -Q update_embed_queue --detach --hostname=update_embeddings_worker@%h
celery -A celery_embed_index.celery worker --loglevel=info --logfile=celery_embed_index.log -P solo -Q embed_index_queue --detach --hostname=embed_index_worker@%h
