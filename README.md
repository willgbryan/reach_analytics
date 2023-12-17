# Reach

## Starting the app

Route to /placeholder1/reach-frontend and run `npm start`. This will launch the main page on port 3000.

Open and run the testing.py file. This will start up the flask app. The flask app will remaining running until it is stopped or a significant error causes a program exit.

## Connecting the backend using Flask

This section will provide a detailed guide on how to connect the backend using Flask. The main files involved in this process are [testing.py](three/testing.py) and [main_page.js](reach-frontend/src/main_page.js).

Flask enables defined methods or classes to act as API's with defined input request formats. In [testing.py](three/testing.py), the following code is used to configure the flask app and declare necessary file paths for functions such as uploading documents:

```python
app = Flask(__name__)
CORS(app)
upload_folder = 'web_upload/datasets'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), upload_folder)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
```

The entirety of the app is defined by:

```python
app = Flask(__name__)

"""App configuration and app.routes"""

if __name__ == '__main__':
    app.run(debug=True)
```

Within the app, defined in [testing.py](three/testing.py), @app.route('/process_prompt', methods=['POST']), @app.route('/upload_files', methods=['POST']), and @app.route('/datasets/<path:filename>', methods=['GET']) define 3 possible request bodies that can be sent to the app via the frontend: [main_page.js](reach-frontend/src/main_page.js). The return

In [main_page.js](reach-frontend/src/main_page.js), the following handles sending the user input value within the text box to the flask app as a POST request:

```javascript
const handleSubmit = async () => {
      setIsGeneratingAnalytics(true);
      const response = await fetch('http://localhost:5000/process_prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: inputValue }),
      });
```

The above code is called within the following div component:

```javascript
 <div className={`${hasSubmitted ? 'p-4' : 'flex-grow flex items-center justify-center'}`}>
          <div className="flex items-center space-x-2">
            {fileNames.length > 0 && (
              <button
                onClick={handleSubmit}
                className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded"
              >
                Submit
              </button>
            )}
```

The return type for each method is defined by the python function being called. In it's current state, return responses are the string body outputs with no additional handling.

This is likely a very complicated implementation that can be refined for increased readability.
