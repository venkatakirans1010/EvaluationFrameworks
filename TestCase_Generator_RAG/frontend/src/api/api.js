const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Generate test cases from Jira ID and uploaded files
 * @param {string} jiraId - Jira issue ID
 * @param {File[]} files - Array of files to upload
 * @returns {Promise<{markdown: string, excel_path: string}>}
 */
export const generateTestCases = async (jiraId, files) => {
  const formData = new FormData();
  formData.append('jira_id', jiraId);
  
  // Append all files
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await fetch(`${API_BASE_URL}/generate_test_cases`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

/**
 * Get download URL for Excel file
 * @param {string} excelPath - Path returned from backend
 * @returns {string} Full URL to download the file
 */
export const getDownloadUrl = (excelPath) => {
  const filename = excelPath.split('/').pop();
  return `${API_BASE_URL}/download/${filename}`;
};

/**
 * Health check endpoint
 * @returns {Promise<{status: string}>}
 */
export const checkHealth = async () => {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('Health check failed');
  }
  return await response.json();
};

