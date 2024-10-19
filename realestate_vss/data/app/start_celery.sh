#!/usr/bin/env sh

if [ -n "$ZSH_VERSION" ]; then
  echo "Running with Zsh"
elif [ -n "$BASH_VERSION" ]; then
  echo "Running with Bash"
else
  echo "Running with another shell, assuming compatibility"
fi

# ps aux | grep 'celery' | awk '{print $2}' | xargs kill -9
# mac: rabbitmqctl list_queues; rabbitmqctl purge_queue embed_index_queue

celery -A celery_unstack.celery worker --loglevel=info --logfile=celery_unstack.log -Q unstack_queue --detach --hostname=unstack_worker@%h
#celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log -P solo -Q embed_queue --detach --hostname=embed_worker@%h 
#celery -A celery_update_embeddings.celery worker --loglevel=info --logfile=celery_update_embeddings.log -P solo -Q update_embed_queue --detach --hostname=update_embeddings_worker@%h
celery -A celery_embed_index.celery worker --loglevel=info --logfile=celery_embed_index.log -P solo -Q embed_index_queue --detach --hostname=embed_index_worker@%h
