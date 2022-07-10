### **Haar Face and Eye Detection Inference API**<br>

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4. Run `start-api-server.bat` (or setup `.vscode`).
5. The API will now be served at `http://127.0.0.1:10010`

**OR**


1. Pull the docker image using `docker pull pchandrasekaran1595/haar-api`
2. Run `docker-run.bat`. 
3. The API will now be served at `http://127.0.0.1:10010`

<br>

**Endpoints**

- `/detect/face`
- `/detect/eye`

<br>