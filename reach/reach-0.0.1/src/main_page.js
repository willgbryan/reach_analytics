import React from 'react';

import { ReactComponent as FolderFill } from 'bootstrap-icons/icons/folder-fill.svg';
import { ReactComponent as GearFill } from 'bootstrap-icons/icons/gear-fill.svg';
import { ReactComponent as FileEarmarkArrowUp} from 'bootstrap-icons/icons/file-earmark-arrow-up.svg';

function UserDiv() {
    const [inputValue, setInputValue] = React.useState('');
    const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);
    const [isDataOpen, setIsDataOpen] = React.useState(false);
    const [isPopupOpen, setIsPopupOpen] = React.useState(false);
    const [fileNames, setFileNames] = React.useState([]);
    const [backendResponse, setBackendResponse] = React.useState('');
    const [hasSubmitted, setHasSubmitted] = React.useState(false);
    const handleInputChange = (event) => {
      setInputValue(event.target.value);
    };
    const handleFileSelection = (event) => {
      const files = event.target.files;
      const names = [...Array.from(files).map(file => file.name)];
      setFileNames(names); // Update the state with the selected file names
    };

    const handleFileUpload = async () => {
      const formData = new FormData();
      const fileInput = document.getElementById('file-input');
      const files = fileInput.files;
  
      // Using a standard for loop to iterate over FileList
      for (let i = 0; i < files.length; i++) {
          formData.append('file', files[i]);
      }
  
      try {
          const response = await fetch('http://localhost:5000/upload_files', {
              method: 'POST',
              body: formData,
          });
  
          if (response.ok) {
              console.log('File uploaded successfully');
              setIsPopupOpen(false); // Close the popup after successful upload
          } else {
              console.error('Upload failed');
          }
      } catch (error) {
          console.error('Error:', error);
      }
    };
  
    const handleButtonClick = () => {
      setIsPopupOpen(true);
    };
    // const handleFileDrop = (event) => {
    //   const files = event.target.files;
    //   const names = [...fileNames, ...Array.from(files).map(file => file.name)];
    //   setFileNames(names);
    //   setIsPopupOpen(false);
    // };
    const toggleSettings = () => {
      setIsSettingsOpen(!isSettingsOpen);
    };
    const toggleData = () => {
      setIsDataOpen(!isDataOpen);
    };
    const handleSubmit = async () => {
      const response = await fetch('http://localhost:5000/process_prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: inputValue }),
      });
      const data = await response.json();
      setBackendResponse(JSON.stringify(data, null, 2));
      setInputValue('');
      setHasSubmitted(true);
    };
    
    return (
      <div className="flex flex-col h-screen">
        <div className="flex justify-between p-4">
          <div></div>
          <div>
            <button onClick={toggleSettings} className="p-2">
              <GearFill className="w-6 h-6" />
            </button>
            {isSettingsOpen && (
              <div className="absolute top-16 right-4 w-64 p-4 bg-white border rounded shadow">
                {/* Settings content goes here */}
              </div>
            )}
          </div>
        </div>
        {hasSubmitted && (
          <div className="flex-grow overflow-auto">
            <pre className="p-4">{backendResponse}</pre>
          </div>
        )}
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
            <input
              type="text"
              value={inputValue}
              onChange={handleInputChange}
              className="border-2 border-gray-300 bg-white h-10 px-5 pr-16 rounded-lg text-sm focus:outline-none flex-grow"
            />
            <button
              onClick={handleButtonClick}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            >
              +
            </button>
          </div>
        </div>
        <div className="flex justify-end p-4">
          <button onClick={toggleData} className="p-2">
            <FolderFill className="w-6 h-6" />
          </button>
          {isDataOpen && (
            <div className="absolute bottom-16 right-4 w-64 p-4 bg-white border rounded shadow">
              <ul>
                {fileNames.map((name, index) => (
                  <li key={index}>{name}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
        {isPopupOpen && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full" id="my-modal">
            <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                <div className="mt-3 text-center">
                    {/* ... existing elements */}
                    <div className="mt-2 px-7 py-3">
                        <input
                            type="file"
                            multiple
                            id="file-input"
                            onChange={handleFileSelection}
                            className="mt-1 block w-full"
                        />
                    </div>
                    <div className="items-center px-4 py-3">
                        <button
                            onClick={handleFileUpload}
                            className="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300"
                        >
                            Upload Files
                        </button>
                        <button
                            id="ok-btn"
                            onClick={() => setIsPopupOpen(false)}
                            className="px-4 py-2 bg-green-500 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-300"
                        >
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )}
      </div>
    );
  }

  export default UserDiv;