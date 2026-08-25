import ReactMarkdown from 'react-markdown';

const ResultTable = ({ markdown, downloadUrl, onDownload, ragStatus }) => {
  if (!markdown) {
    return null;
  }

  return (
    <div className="mt-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-xl font-semibold text-gray-800">Generated Test Cases</h2>
              {ragStatus && (
                <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  ragStatus.enabled 
                    ? 'bg-green-100 text-green-800 border border-green-300' 
                    : ragStatus.index_built && ragStatus.files_uploaded > 0
                    ? 'bg-yellow-100 text-yellow-800 border border-yellow-300'
                    : 'bg-gray-100 text-gray-600 border border-gray-300'
                }`}>
                  {ragStatus.enabled ? (
                    <>
                      <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      RAG Enabled
                    </>
                  ) : ragStatus.index_built && ragStatus.files_uploaded > 0 ? (
                    <>
                      <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      RAG No Match
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4 mr-1.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                      RAG Disabled
                    </>
                  )}
                </div>
              )}
            </div>
            {ragStatus && (
              <p className={`text-sm ${
                ragStatus.enabled 
                  ? 'text-green-700' 
                  : ragStatus.index_built && ragStatus.files_uploaded > 0
                  ? 'text-yellow-700'
                  : 'text-gray-600'
              }`}>
                {ragStatus.message}
                {ragStatus.enabled && ragStatus.context_length > 0 && (
                  <span className="ml-2 text-xs text-gray-500">
                    ({ragStatus.context_length.toLocaleString()} chars retrieved)
                  </span>
                )}
              </p>
            )}
          </div>
          {downloadUrl && (
            <a
              href={downloadUrl}
              download
              onClick={onDownload}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors duration-200 font-medium ml-4"
            >
              Download Excel
            </a>
          )}
        </div>
        <div className="prose max-w-none overflow-x-auto">
          <ReactMarkdown
            components={{
              table: ({ node, ...props }) => (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 border border-gray-300" {...props} />
                </div>
              ),
              thead: ({ node, ...props }) => (
                <thead className="bg-gray-50" {...props} />
              ),
              tbody: ({ node, ...props }) => (
                <tbody className="bg-white divide-y divide-gray-200" {...props} />
              ),
              th: ({ node, ...props }) => (
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border border-gray-300" {...props} />
              ),
              td: ({ node, ...props }) => (
                <td className="px-4 py-3 text-sm text-gray-900 border border-gray-300 whitespace-pre-wrap" {...props} />
              ),
            }}
          >
            {markdown}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default ResultTable;

