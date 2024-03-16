#!/bin/zsh

celery -A celery_unstack.celery worker --loglevel=info --logfile=celery_unstack.log -Q unstack_queue --detach --hostname=unstack_worker@%h
celery -A celery_embed.celery worker --loglevel=info --logfile=celery_embed.log -P solo -Q embed_queue --detach --hostname=embed_worker@%h 
