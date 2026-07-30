"""add_url_and_summary_to_news

Revision ID: 0e836bfecc39
Revises: f50a04ab5936
Create Date: 2026-07-25 18:37:48.920304+00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0e836bfecc39'
down_revision: Union[str, Sequence[str], None] = 'f50a04ab5936'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('news_articles', sa.Column('url', sa.String(), nullable=True))
    op.add_column('news_articles', sa.Column('summary', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('news_articles', 'summary')
    op.drop_column('news_articles', 'url')
