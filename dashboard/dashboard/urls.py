from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="dashboard"),
    path("events/<int:event_id>/review/", views.review_event, name="review-event"),
    path("snapshots/<path:relative_path>", views.snapshot, name="snapshot"),
]
