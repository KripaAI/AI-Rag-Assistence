from src.config import settings


def test_settings_load():
    assert settings.pinecone_index
    assert settings.openai_chat_model
