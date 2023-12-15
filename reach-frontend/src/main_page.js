import React from 'react';
import AceEditor from 'react-ace';
import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/theme-monokai';
import { ReactComponent as FolderFill } from 'bootstrap-icons/icons/folder-fill.svg';
import { ReactComponent as GearFill } from 'bootstrap-icons/icons/gear-fill.svg';
import { ReactComponent as FileEarmarkArrowUp} from 'bootstrap-icons/icons/file-earmark-arrow-up.svg';
import { ReactComponent as XCircleFill } from 'bootstrap-icons/icons/x-circle-fill.svg';
import { ReactComponent as GridFill } from 'bootstrap-icons/icons/grid-fill.svg'; // Import the grid icon
import axios from 'axios'; // Import axios for making HTTP requests

function UserDiv() {
  const [inputValue, setInputValue] = React.useState('');
  const [isSettingsOpen, setIsSettingsOpen] = React.useState(false);
  const [isDataOpen, setIsDataOpen] = React.useState(false);
  const [isPopupOpen, setIsPopupOpen] = React.useState(false);
  const [fileNames, setFileNames] = React.useState([]);
  const [hasSubmitted, setHasSubmitted] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState(false);
  const [backendResponses, setBackendResponses] = React.useState([]);
  const [backendResponse, setBackendResponse] = React.useState('');
  const [isIDEPaneOpen, setIsIDEPaneOpen] = React.useState(false);
  const [validatedCode, setValidatedCode] = React.useState('');
  const [isGeneratingAnalytics, setIsGeneratingAnalytics] = React.useState(false);
  const [isDataPreviewOpen, setIsDataPreviewOpen] = React.useState(false);
  const [dataPreview, setDataPreview] = React.useState('');

    const toggleIDEPane = () => {
        setIsIDEPaneOpen(!isIDEPaneOpen);
    };

    const handleInputChange = (event) => {
      setInputValue(event.target.value);
    };
    const handleFileSelection = (event) => {
      const files = event.target.files;
      const names = [...Array.from(files).map(file => file.name)];
      setFileNames(names);
    };

    const handleFileUpload = async () => {
      setIsPopupOpen(false);
      setIsLoading(true);
      const formData = new FormData();
      const fileInput = document.getElementById('file-input');
      const files = fileInput.files;
  
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
          } else {
              console.error('Upload failed');
          }
      } catch (error) {
          console.error('Error:', error);
      }
      setIsLoading(false);
  };
  
    const handleButtonClick = () => {
      setIsPopupOpen(true);
    };
    
    const toggleSettings = () => {
      setIsSettingsOpen(!isSettingsOpen);
    };
    const toggleData = () => {
      setIsDataOpen(!isDataOpen);
    };
    const handleSubmit = async () => {
      setIsGeneratingAnalytics(true);
      const response = await fetch('http://localhost:5000/process_prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: inputValue }),
      });
      const data = await response.json();
        setBackendResponses(prevResponses => [...prevResponses, JSON.stringify(data.codeOutput, null, 2)]);
        setValidatedCode(data.validatedCode);
        setInputValue('');
        setHasSubmitted(true);
        setIsGeneratingAnalytics(false);
    };      
    
    const removeFile = (index) => {
      setFileNames(fileNames.filter((_, i) => i !== index));
    };

    const handleDataPreview = async () => {
      setIsDataPreviewOpen(true);
      try {
        const response = await axios.get('http://localhost:5000/datasets/aggregated_data.csv');        setDataPreview(response.data);
      } catch (error) {
        console.error('Error:', error);
      }
    };

    return (
      <div className="flex flex-col h-screen">
        <button onClick={toggleIDEPane} style={{ position: 'absolute', top: 0, left: 0 }}>
                Code Pane
        </button>
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
            <button onClick={toggleData} className="p-2">
              <FolderFill className="w-6 h-6" />
            </button>
            {isDataOpen && (
            <div className="absolute top-16 right-4 w-64 p-4 bg-white border rounded shadow">
                <ul>
                {fileNames.map((name, index) => (
                    <li key={index}>
                    {name}
                    <button onClick={() => removeFile(index)} className="p-1">
                        <XCircleFill className="w-4 h-4" />
                    </button>
                    </li>
                ))}
                </ul>
            </div>
            )}
            <button onClick={handleDataPreview} className="p-2">
              <GridFill className="w-6 h-6" />
            </button>
            {isDataPreviewOpen && (
            <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full flex items-center justify-center">
                <div className="relative mx-auto p-5 border shadow-lg rounded-md bg-white" style={{ maxHeight: '900px', maxWidth: '900px', overflowY: 'auto' }}>
                <div className="flex justify-between items-center">
                    <div></div> {/* Empty div for balancing flex space-between */}
                    <h2>Aggregated Dataset</h2>
                    <button onClick={() => setIsDataPreviewOpen(false)} className="p-1">
                    <XCircleFill className="w-4 h-4" />
                    </button>
                </div>
                <pre>{dataPreview}</pre>
                </div>
            </div>
            )}
          </div>
        </div>
        {isLoading && (
            <div className="loading-container" style={{ textAlign: 'center' }}>
                <p>Aggregating supplied data, this may take a few minutes...</p> {/* replace this with a spinner or any other loading indicator */}
            </div>
        )}
        {isGeneratingAnalytics && (
            <div className="loading-container" style={{ textAlign: 'center' }}>
                <p>Generating analytics...</p>
            </div>
        )}
        {hasSubmitted && (
                <div className="flex-grow overflow-auto" style={{ maxHeight: '50vh' }}>
                    <div style={{ width: '50%', margin: '0 auto' }}>
                        {backendResponses.map((response, index) => (
                            <pre key={index} className="p-4" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                                {response}
                            </pre>
                        ))}
                    </div>
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
    <div style={{ 
            position: 'absolute', 
            top: '40px', 
            left: 0, 
            width: '100%', 
            maxHeight: isIDEPaneOpen ? '500px' : '0',
            overflow: 'hidden',
            transition: 'max-height 0.3s ease-in-out'
        }}>
            <AceEditor
                mode="python"
                theme="monokai"
                name="python-editor"
                value={validatedCode}
                editorProps={{ $blockScrolling: true }}
                style={{ width: '100%', height: '500px' }}
            />
        </div>
      </div>
    );
  }

export default UserDiv;