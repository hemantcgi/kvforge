from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3

from connectors.base import SourceFile


@dataclass
class S3Connector:
    """S3 source connector implementing SourceConnector protocol.

    When local_mirror_path is set, reads from local filesystem instead of S3.
    Otherwise, uses boto3 to list and download objects from S3.
    """
    bucket: str
    prefix: str = ""
    region: str = "us-east-1"
    access_key_id: str = ""
    secret_access_key: str = ""
    local_mirror_path: str = ""

    def __post_init__(self):
        if not self.local_mirror_path:
            self._client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=self.access_key_id if self.access_key_id else None,
                aws_secret_access_key=self.secret_access_key if self.secret_access_key else None,
            )
        else:
            self._client = None

    def list_files(self) -> list[SourceFile]:
        """List files from S3 (or local mirror if configured)."""
        if self.local_mirror_path:
            return self._list_files_local()
        return self._list_files_s3()

    def _list_files_local(self) -> list[SourceFile]:
        """List files from local mirror path."""
        root = Path(self.local_mirror_path)
        files = []

        if not root.exists():
            return files

        for file_path in root.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                # Convert timestamp to UTC datetime
                modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

                files.append(
                    SourceFile(
                        id=str(file_path.relative_to(root)),
                        name=file_path.name,
                        path=str(file_path),
                        size=stat.st_size,
                        modified_at=modified_at,
                    )
                )

        return files

    def _list_files_s3(self) -> list[SourceFile]:
        """List files from S3 bucket using paginator."""
        files = []
        paginator = self._client.get_paginator("list_objects_v2")

        page_iterator = paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.prefix,
        )

        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]

                # Skip directory-like entries
                if key.endswith("/"):
                    continue

                # Extract filename from full key, stripping prefix
                if self.prefix and key.startswith(self.prefix):
                    relative_key = key[len(self.prefix):]
                else:
                    relative_key = key

                file_name = Path(relative_key).name

                files.append(
                    SourceFile(
                        id=key,  # Full key is the stable unique ID
                        name=file_name,  # Just the filename
                        path=key,
                        size=obj["Size"],
                        modified_at=obj["LastModified"],
                    )
                )

        return files

    def download(self, file: SourceFile) -> bytes:
        """Download file content."""
        if self.local_mirror_path:
            return self._download_local(file)
        return self._download_s3(file)

    def _download_local(self, file: SourceFile) -> bytes:
        """Download file from local mirror path."""
        file_path = Path(file.path)
        return file_path.read_bytes()

    def _download_s3(self, file: SourceFile) -> bytes:
        """Download file from S3."""
        response = self._client.get_object(Bucket=self.bucket, Key=file.id)
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def get_modified_at(self, file: SourceFile) -> datetime:
        """Return modification time of the file."""
        return file.modified_at

    def supports_delta(self) -> bool:
        """S3 connector does not support delta sync."""
        return False

    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]:
        """Return all files and empty token (delta not supported)."""
        return self.list_files(), ""
