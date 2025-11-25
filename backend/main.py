from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import os
import numpy as np
import logging
import cv as cv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Process Image", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-image/")
async def upload_image(
    file: UploadFile = File(..., description="Изображение для обработки"),
    risk_threshold: int = 60
):
    """
    Загружает изображение и возвращает результат обработки.
    
    Args:
        file: Изображение в формате JPG/PNG
        risk_threshold: Порог для обнаружения рисков (по умолчанию 60)
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="Файл должен быть изображением. Поддерживаемые форматы: JPG, PNG"
            )
        
        allowed_extensions = {'.jpg', '.jpeg', '.png'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Неподдерживаемый формат файла. Разрешены: {', '.join(allowed_extensions)}"
            )
        
        max_size = 10 * 1024 * 1024
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой. Максимальный размер: {max_size // (1024 * 1024)}MB"
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400, 
                detail="Файл не может быть пустым"
            )
        
        logger.info(f"Начата обработка изображения: {file.filename}")
        
        result = cv.process_image(file, risk_threshold)
        
        logger.info(f"Изображение {file.filename} успешно обработано. Результат: {result['value']:.2f}")
        
        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "risk_threshold": risk_threshold,
                "result": result,
                "message": "Изображение успешно обработано"
            }
        )
        
    except HTTPException:
        raise
        
    except ValueError as e:
        logger.warning(f"Ошибка обработки изображения {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Ошибка обработки изображения: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при обработке файла {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Внутренняя ошибка сервера при обработке изображения"
        )

@app.get("/")
async def root():
    """
    Корневой endpoint для проверки работы API.
    """
    return {"message": "Image Processing API работает корректно"}

@app.get("/health")
async def health_check():
    """
    Endpoint для проверки здоровья сервиса.
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )