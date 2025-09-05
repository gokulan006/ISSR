"""
URL configuration for mieapi project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ragapi import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/rag/modular/train', views.rag_modular_train),
    path('api/rag/modular/query', views.rag_modular_query),
    path('api/rag/model/init', views.rag_model_init),
    path('api/rag/model/similar', views.rag_model_similar),
    path('api/rag/model/classify', views.rag_model_classify),
    path('api/rag/model/classify_json', views.rag_model_classify_json),
    path('api/agent/run', views.agent_run),
    path('api/agent/catalog/ingest', views.agent_catalog_ingest),
    path('api/agent/catalog/list', views.agent_catalog_list),
]
