import axios from 'axios';

const BASE_URL = 'http://localhost:5000';

export const analyzeXray = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  const response = await axios.post(`${BASE_URL}/analyze`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const submitFeedback = async (feedbackData) => {
  const response = await axios.post(`${BASE_URL}/feedback`, feedbackData);
  return response.data;
};