"""
SentinelSite — S3 Storage Service
Handles audio clips, frames, models, admin images.
Supports real AWS S3 and local MinIO (dev mode).
All paths are namespaced by site_id for strict isolation.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import settings

log = logging.getLogger(__name__)


class StorageService:
    """
    Singleton storage client. Import `storage` from this module.
    All bucket names come from settings — never hardcode.
    """

    def __init__(self) -> None:
        kwargs: dict = {
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            "region_name": settings.AWS_REGION,
            "config": Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=20,
            ),
        }
        if settings.USE_MINIO:
            kwargs["endpoint_url"] = settings.MINIO_ENDPOINT

        self._client = boto3.client("s3", **kwargs)
        self._resource = boto3.resource("s3", **kwargs)
        log.info(
            f"StorageService ready — "
            f"{'MinIO @ ' + settings.MINIO_ENDPOINT if settings.USE_MINIO else 'AWS S3'}"
        )

    # ── Key builders ──────────────────────────────────────────────────────────

    @staticmethod
    def audio_key(site_id: str, event_id: str) -> str:
        """s3://sentinelsite-audio/{site_id}/events/{event_id}/audio.wav"""
        return f"{site_id}/events/{event_id}/audio.wav"

    @staticmethod
    def frame_key(site_id: str, event_id: str) -> str:
        """s3://sentinelsite-frames/{site_id}/events/{event_id}/frame.jpg"""
        return f"{site_id}/events/{event_id}/frame.jpg"

    @staticmethod
    def model_key(site_id: str, model_type: str, version: int) -> str:
        """s3://sentinelsite-models/{site_id}/{model_type}/v{version}/model.tflite"""
        return f"{site_id}/{model_type}/v{version}/model.tflite"

    @staticmethod
    def model_checkpoint_key(site_id: str, model_type: str, version: int) -> str:
        """PyTorch .pt checkpoint for rollback"""
        return f"{site_id}/{model_type}/v{version}/checkpoint.pt"

    @staticmethod
    def admin_image_key(site_id: str, class_label: str, filename: str) -> str:
        """s3://sentinelsite-admin/{site_id}/{class_label}/{uuid}_{filename}"""
        uid = str(uuid4())[:8]
        safe_label = class_label.replace(" ", "_").lower()
        return f"{site_id}/{safe_label}/{uid}_{filename}"

    @staticmethod
    def document_key(site_id: str, filename: str) -> str:
        """s3://sentinelsite-admin/{site_id}/documents/{uuid}_{filename}"""
        uid = str(uuid4())[:8]
        return f"{site_id}/documents/{uid}_{filename}"

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload_audio(
        self,
        site_id: str,
        event_id: str,
        data: bytes | BinaryIO,
        content_type: str = "audio/wav",
    ) -> str:
        """Upload 30s audio clip. Returns S3 key."""
        key = self.audio_key(site_id, event_id)
        self._upload(
            bucket=settings.S3_BUCKET_AUDIO,
            key=key,
            data=data,
            content_type=content_type,
            extra_args={
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "site_id": site_id,
                    "event_id": event_id,
                    "uploaded_at": datetime.utcnow().isoformat(),
                },
            },
        )
        log.info(f"Audio uploaded: {key}")
        return key

    def upload_frame(
        self,
        site_id: str,
        event_id: str,
        data: bytes | BinaryIO,
    ) -> str:
        """Upload event frame JPEG. Returns S3 key."""
        key = self.frame_key(site_id, event_id)
        self._upload(
            bucket=settings.S3_BUCKET_FRAMES,
            key=key,
            data=data,
            content_type="image/jpeg",
            extra_args={
                "ServerSideEncryption": "AES256",
                "Metadata": {"site_id": site_id, "event_id": event_id},
            },
        )
        log.info(f"Frame uploaded: {key}")
        return key

    def upload_model(
        self,
        site_id: str,
        model_type: str,
        version: int,
        tflite_data: bytes,
        pytorch_data: bytes | None = None,
    ) -> tuple[str, str | None]:
        """
        Upload TFLite model (and optionally the PyTorch checkpoint).
        Returns (tflite_key, checkpoint_key | None).
        """
        tflite_key = self.model_key(site_id, model_type, version)
        self._upload(
            bucket=settings.S3_BUCKET_MODELS,
            key=tflite_key,
            data=tflite_data,
            content_type="application/octet-stream",
            extra_args={
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "site_id": site_id,
                    "model_type": model_type,
                    "version": str(version),
                },
            },
        )

        ckpt_key = None
        if pytorch_data is not None:
            ckpt_key = self.model_checkpoint_key(site_id, model_type, version)
            self._upload(
                bucket=settings.S3_BUCKET_MODELS,
                key=ckpt_key,
                data=pytorch_data,
                content_type="application/octet-stream",
                extra_args={"ServerSideEncryption": "AES256"},
            )

        log.info(f"Model uploaded: {tflite_key}")
        return tflite_key, ckpt_key

    def upload_admin_image(
        self,
        site_id: str,
        class_label: str,
        filename: str,
        data: bytes | BinaryIO,
    ) -> str:
        """Upload admin training image. Returns S3 key."""
        key = self.admin_image_key(site_id, class_label, filename)
        self._upload(
            bucket=settings.S3_BUCKET_ADMIN_IMAGES,
            key=key,
            data=data,
            content_type="image/jpeg",
            extra_args={
                "ServerSideEncryption": "AES256",
                "Metadata": {
                    "site_id": site_id,
                    "class_label": class_label,
                },
            },
        )
        return key

    def upload_document(
        self,
        site_id: str,
        filename: str,
        data: bytes | BinaryIO,
        content_type: str = "application/pdf",
    ) -> str:
        """Upload site document PDF. Returns S3 key."""
        key = self.document_key(site_id, filename)
        self._upload(
            bucket=settings.S3_BUCKET_ADMIN_IMAGES,
            key=key,
            data=data,
            content_type=content_type,
            extra_args={"ServerSideEncryption": "AES256"},
        )
        return key

    # ── Download ──────────────────────────────────────────────────────────────

    def download_bytes(self, bucket: str, key: str) -> bytes:
        """Download object as bytes. Used by Celery workers."""
        try:
            obj = self._client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()
        except ClientError as e:
            log.error(f"S3 download failed: {bucket}/{key} — {e}")
            raise

    def download_audio(self, key: str) -> bytes:
        return self.download_bytes(settings.S3_BUCKET_AUDIO, key)

    def download_frame(self, key: str) -> bytes:
        return self.download_bytes(settings.S3_BUCKET_FRAMES, key)

    def download_model(self, key: str) -> bytes:
        return self.download_bytes(settings.S3_BUCKET_MODELS, key)

    def download_document(self, key: str) -> bytes:
        return self.download_bytes(settings.S3_BUCKET_ADMIN_IMAGES, key)

    def download_to_file(self, bucket: str, key: str, local_path: str) -> None:
        """Stream large file directly to disk (models)."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(bucket, key, local_path)
        log.info(f"Downloaded {bucket}/{key} → {local_path}")

    # ── Presigned URLs ────────────────────────────────────────────────────────

    def presigned_audio_url(self, key: str, expiry: int | None = None) -> str:
        return self._presign(
            settings.S3_BUCKET_AUDIO,
            key,
            expiry or settings.S3_PRESIGNED_URL_EXPIRY,
        )

    def presigned_frame_url(self, key: str, expiry: int | None = None) -> str:
        return self._presign(
            settings.S3_BUCKET_FRAMES,
            key,
            expiry or settings.S3_PRESIGNED_URL_EXPIRY,
        )

    def presigned_model_url(self, key: str, expiry: int | None = None) -> str:
        """OTA model download URL — 1h default, devices cache aggressively."""
        return self._presign(
            settings.S3_BUCKET_MODELS,
            key,
            expiry or settings.S3_PRESIGNED_URL_EXPIRY,
        )

    def _presign(self, bucket: str, key: str, expiry: int) -> str:
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expiry,
            )
        except ClientError as e:
            log.error(f"Presign failed: {bucket}/{key} — {e}")
            raise

    # ── Existence check ───────────────────────────────────────────────────────

    def exists(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, bucket: str, key: str) -> None:
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
            log.info(f"Deleted: {bucket}/{key}")
        except ClientError as e:
            log.error(f"Delete failed: {bucket}/{key} — {e}")
            raise

    # ── Bucket bootstrap (local dev) ──────────────────────────────────────────

    def ensure_buckets_exist(self) -> None:
        """
        Create buckets if they don't exist.
        Only relevant for local MinIO setup — real AWS uses Terraform.
        """
        buckets = [
            settings.S3_BUCKET_AUDIO,
            settings.S3_BUCKET_FRAMES,
            settings.S3_BUCKET_MODELS,
            settings.S3_BUCKET_ADMIN_IMAGES,
        ]
        for bucket in buckets:
            try:
                self._client.head_bucket(Bucket=bucket)
            except ClientError:
                self._client.create_bucket(Bucket=bucket)
                log.info(f"Created bucket: {bucket}")

    # ── Health check ──────────────────────────────────────────────────────────

    def check_health(self) -> dict:
        try:
            self._client.head_bucket(Bucket=settings.S3_BUCKET_AUDIO)
            return {"status": "ok", "storage": "s3"}
        except Exception as e:
            return {"status": "error", "storage": "s3", "detail": str(e)}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _upload(
        self,
        bucket: str,
        key: str,
        data: bytes | BinaryIO,
        content_type: str,
        extra_args: dict | None = None,
    ) -> None:
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        args = {"ContentType": content_type}
        if extra_args:
            args.update(extra_args)
        try:
            self._client.upload_fileobj(data, bucket, key, ExtraArgs=args)
        except ClientError as e:
            log.error(f"Upload failed: {bucket}/{key} — {e}")
            raise


# ── Module-level singleton ────────────────────────────────────────────────────
storage = StorageService()
