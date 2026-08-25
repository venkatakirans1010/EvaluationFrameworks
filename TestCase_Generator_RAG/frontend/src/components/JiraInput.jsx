import { useState } from 'react';

const JiraInput = ({ value, onChange, disabled }) => {
  return (
    <div className="mb-4">
      <label htmlFor="jira-id" className="block text-sm font-medium text-gray-700 mb-2">
        Jira Issue ID
      </label>
      <input
        id="jira-id"
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder="e.g., PROJ-123"
        className="w-full px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
      />
    </div>
  );
};

export default JiraInput;

