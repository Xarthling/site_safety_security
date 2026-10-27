from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="dashboard"),
    path("cameras/", views.cameras, name="cameras"),
    path("cameras/add/", views.camera_add, name="camera-add"),
    path("cameras/<str:camera_id>/", views.camera_edit, name="camera-edit"),
    path("cameras/<str:camera_id>/toggle/", views.camera_toggle, name="camera-toggle"),
    path("cameras/<str:camera_id>/test/", views.camera_test, name="camera-test"),
    path("cameras/<str:camera_id>/replay/", views.camera_replay, name="camera-replay"),
    path("cameras/<str:camera_id>/live/", views.camera_live, name="camera-live"),
    path("cameras/<str:camera_id>/stream/", views.camera_stream, name="camera-stream"),
    path("cameras/<str:camera_id>/rules/", views.camera_rules, name="camera-rules"),
    path("cameras/<str:camera_id>/video/", views.camera_video, name="camera-video"),
    path("cameras/<str:camera_id>/frame/", views.camera_frame, name="camera-frame"),
    path("events/<int:event_id>/review/", views.review_event, name="review-event"),
    path("snapshots/<path:relative_path>", views.snapshot, name="snapshot"),
]
