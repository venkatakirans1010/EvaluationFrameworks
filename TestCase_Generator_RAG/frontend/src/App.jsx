import { useState } from 'react';
import JiraInput from './components/JiraInput';
import FileUploader from './components/FileUploader';
import ResultTable from './components/ResultTable';
import { generateTestCases, getDownloadUrl } from './api/api';

function App() {
  const [jiraId, setJiraId] = useState('');
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleGenerate = async () => {
    // Validation
    if (!jiraId.trim()) {
      setError('Please enter a Jira ID');
      return;
    }

    setError(null);
    setLoading(true);
    setResult(null);

    try {
      const response = await generateTestCases(jiraId.trim(), files);
      setResult({
        markdown: response.markdown,
        excelPath: response.excel_path,
        ragStatus: response.rag_status || {
          enabled: false,
          files_uploaded: 0,
          context_length: 0,
          message: 'RAG status not available'
        },
      });
    } catch (err) {
      setError(err.message || 'Failed to generate test cases. Please try again.');
      console.error('Error generating test cases:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    // Optional: Track download event
    console.log('Downloading Excel file:', result.excelPath);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Test Case Generator
          </h1>
          <p className="text-gray-600">
            Generate comprehensive test cases from Jira stories and supporting documents
          </p>
        </header>

        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <JiraInput value={jiraId} onChange={setJiraId} disabled={loading} />
          <FileUploader
            files={files}
            onFilesChange={setFiles}
            disabled={loading}
          />

          {/* RAG Status Indicator */}
          {files.length > 0 && (
            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-md">
              <div className="flex items-center">
                <svg className="w-5 h-5 text-blue-600 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                  <path fillRule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-blue-800">
                    RAG will be enabled
                  </p>
                  <p className="text-xs text-blue-600">
                    {files.length} document{files.length !== 1 ? 's' : ''} will be used for context retrieval
                  </p>
                </div>
              </div>
            </div>
          )}

          {files.length === 0 && (
            <div className="mb-4 p-3 bg-gray-50 border border-gray-200 rounded-md">
              <div className="flex items-center">
                <svg className="w-5 h-5 text-gray-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="text-sm font-medium text-gray-700">
                    RAG will be disabled
                  </p>
                  <p className="text-xs text-gray-500">
                    Upload documents to enable RAG context retrieval
                  </p>
                </div>
              </div>
            </div>
          )}

          <button
            onClick={handleGenerate}
            disabled={loading || !jiraId.trim()}
            className="w-full mt-6 px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200 font-medium text-lg"
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Generating...
              </span>
            ) : (
              'Generate Test Cases'
            )}
          </button>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg
                    className="h-5 w-5 text-red-400"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">Error</h3>
                  <div className="mt-2 text-sm text-red-700">{error}</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {result && (
          <ResultTable
            markdown={result.markdown}
            downloadUrl={getDownloadUrl(result.excelPath)}
            onDownload={handleDownload}
            ragStatus={result.ragStatus}
          />
        )}
      </div>
    </div>
  );
}

export default App;

