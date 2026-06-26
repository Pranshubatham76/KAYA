import React, { useState } from 'react';
import { UploadCloud, CheckCircle } from 'lucide-react';
import axios from 'axios';

export const AdminTraining: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>('');

  const handleUpload = async () => {
    if (!file) return;
    setStatus('Uploading...');
    const formData = new FormData();
    formData.append('file', file);
    try {
      // Mocked endpoint for admin image upload
      // await axios.post('http://127.0.0.1:8000/api/v1/training/admin_images', formData);
      setTimeout(() => {
        setStatus('Uploaded successfully.');
        setFile(null);
      }, 1000);
    } catch (err) {
      setStatus('Upload failed.');
    }
  };

  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <h2 className="text-xl font-bold mb-4 text-white flex items-center gap-2">
        <UploadCloud className="text-primary" /> Admin Model Retraining
      </h2>
      <p className="text-gray-400 text-sm mb-6">
        Upload site-specific imagery (PPE, custom machinery) to fine-tune the Visual MobileNet edge model.
      </p>
      
      <div className="flex-1 border-2 border-dashed border-white/20 rounded-xl flex flex-col items-center justify-center p-6 bg-background/30 transition hover:border-primary/50">
        <UploadCloud size={48} className="text-gray-500 mb-4" />
        <input 
          type="file" 
          onChange={(e) => setFile(e.target.files?.[0] || null)} 
          className="text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary/20 file:text-primary hover:file:bg-primary/30 cursor-pointer"
          accept="image/*,.zip"
        />
        {file && (
          <button onClick={handleUpload} className="mt-6 bg-primary hover:bg-blue-600 text-white px-6 py-2 rounded-lg font-medium transition shadow-lg shadow-primary/20">
            Upload & Train
          </button>
        )}
        {status && (
          <div className="mt-4 text-sm font-medium text-accent flex items-center gap-2">
            <CheckCircle size={16} /> {status}
          </div>
        )}
      </div>
    </div>
  );
};
