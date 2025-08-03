@echo off
echo Checking library versions for Agent System project...
echo ================================================= > library_versions.txt
echo Agent System - Library Versions >> library_versions.txt
echo Generated on: %date% %time% >> library_versions.txt
echo ================================================= >> library_versions.txt
echo. >> library_versions.txt

echo Python Version: >> library_versions.txt
python --version >> library_versions.txt 2>&1
echo. >> library_versions.txt

echo Pip Version: >> library_versions.txt
pip --version >> library_versions.txt 2>&1
echo. >> library_versions.txt

echo Checking ALL external libraries... >> library_versions.txt
echo --------------------------------- >> library_versions.txt

echo === Core LangChain packages === >> library_versions.txt
echo langchain: >> library_versions.txt
pip show langchain >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo langchain-core: >> library_versions.txt
pip show langchain-core >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo langchain-openai: >> library_versions.txt
pip show langchain-openai >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo langchain-community: >> library_versions.txt
pip show langchain-community >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === Data processing libraries === >> library_versions.txt
echo pandas: >> library_versions.txt
pip show pandas >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo numpy: >> library_versions.txt
pip show numpy >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo datasets: >> library_versions.txt
pip show datasets >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === Image processing === >> library_versions.txt
echo Pillow: >> library_versions.txt
pip show Pillow >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === UI and progress === >> library_versions.txt
echo tqdm: >> library_versions.txt
pip show tqdm >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === HTTP and API libraries === >> library_versions.txt
echo requests: >> library_versions.txt
pip show requests >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo openai: >> library_versions.txt
pip show openai >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === HuggingFace ecosystem === >> library_versions.txt
echo transformers: >> library_versions.txt
pip show transformers >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo tokenizers: >> library_versions.txt
pip show tokenizers >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo huggingface-hub: >> library_versions.txt
pip show huggingface-hub >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === Async and concurrency === >> library_versions.txt
echo aiohttp: >> library_versions.txt
pip show aiohttp >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === Data validation === >> library_versions.txt
echo pydantic: >> library_versions.txt
pip show pydantic >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo === Additional common dependencies === >> library_versions.txt
echo PyYAML: >> library_versions.txt
pip show PyYAML >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo python-dotenv: >> library_versions.txt
pip show python-dotenv >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo typing-extensions: >> library_versions.txt
pip show typing-extensions >> library_versions.txt 2>&1 || echo Not installed >> library_versions.txt
echo. >> library_versions.txt

echo ================================= >> library_versions.txt
echo Version check complete! >> library_versions.txt
echo Results saved to library_versions.txt

echo.
echo Library version check complete!
echo Results have been saved to library_versions.txt
pause