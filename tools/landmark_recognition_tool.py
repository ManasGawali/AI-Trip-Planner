import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.landmark_cnn import LandmarkPredictor
from typing import List
from langchain.tools import tool

class LandmarkRecognitionTool:
    def __init__(self):
        # Initialize to None - lazy loading
        self.predictor = None
        self.model_loaded = False
        self.landmark_recognition_tool_list = self._setup_tools()
    
    def _lazy_load_predictor(self):
        """Load the predictor only when needed"""
        if self.predictor is None:
            try:
                self.predictor = LandmarkPredictor(
                    model_path='models/saved_models/landmark_model_best.pth',
                    landmark_info_path='data/landmarks_compact/landmark_info.json'
                )
                self.model_loaded = True
                print("✓ LandmarkPredictor loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading landmark predictor: {e}")
                self.model_loaded = False

    def _recognize_landmark(self, image_path: str, confidence_threshold: float = 0.3) -> dict:
        """Internal method to recognize landmark"""
        # CALL THE LAZY LOADER HERE!
        self._lazy_load_predictor()
        
        if not self.model_loaded:
            return {
                'success': False,
                'message': 'Landmark recognition model not loaded. Please train the model first.'
            }
        
        try:
            results = self.predictor.predict(image_path, top_k=3)
            
            if not results or results[0]['confidence'] < confidence_threshold * 100:
                return {
                    'success': False,
                    'message': 'Could not identify landmark with sufficient confidence. Try a clearer image.'
                }
            
            top_result = results[0]
            
            return {
                'success': True,
                'landmark_name': top_result['landmark'],
                'city': top_result['city'],
                'country': top_result['country'],
                'latitude': top_result['latitude'],
                'longitude': top_result['longitude'],
                'confidence': top_result['confidence'],
                'all_predictions': results
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Error during prediction: {str(e)}'
            }
    
    def _setup_tools(self) -> List:
        """Setup all tools for landmark recognition"""
        
        @tool
        def identify_landmark_from_image(image_path: str) -> str:
            """
            Identify a landmark from an image file path.
            Returns the landmark name, location, and confidence level.
            
            Args:
                image_path: Path to the landmark image file
            
            Returns:
                String with landmark information including name, city, country, and confidence
            """
            result = self._recognize_landmark(image_path, confidence_threshold=0.3)
            
            if not result['success']:
                return f"Error: {result['message']}"
            
            # Format the response for the agent
            response = f"Identified Landmark: {result['landmark_name']}\n"
            response += f"Location: {result['city']}, {result['country']}\n"
            response += f"Coordinates: {result['latitude']}, {result['longitude']}\n"
            response += f"Confidence: {result['confidence']:.1f}%\n"
            
            if len(result['all_predictions']) > 1:
                response += "\nOther possible landmarks:\n"
                for pred in result['all_predictions'][1:]:
                    response += f"  - {pred['landmark']} ({pred['city']}, {pred['country']}) - {pred['confidence']:.1f}%\n"
            
            return response
        
        @tool
        def recognize_landmark_for_trip_planning(image_path: str) -> dict:
            """
            Recognize a landmark from an image and return structured data for trip planning.
            This tool returns detailed information that can be used to plan a trip.
            
            Args:
                image_path: Path to the landmark image file
            
            Returns:
                Dictionary with landmark details suitable for trip planning
            """
            result = self._recognize_landmark(image_path, confidence_threshold=0.3)
            
            if not result['success']:
                return {
                    'error': result['message'],
                    'suggestion': 'Please provide a clearer image of a famous landmark.'
                }
            
            return {
                'landmark': result['landmark_name'],
                'city': result['city'],
                'country': result['country'],
                'latitude': result['latitude'],
                'longitude': result['longitude'],
                'confidence': f"{result['confidence']:.1f}%",
                'trip_planning_query': f"Plan a trip to {result['landmark_name']} in {result['city']}, {result['country']}"
            }
        
        # Return list of tools
        return [
            identify_landmark_from_image,
            recognize_landmark_for_trip_planning
        ]


# Standalone function for direct use (backwards compatible)
def landmark_recognition_tool(image_path: str) -> dict:
    """
    Standalone function for landmark recognition.
    Used for direct API calls without LangChain.
    
    Args:
        image_path: Path to the landmark image
    
    Returns:
        Dictionary with recognition results
    """
    tool = LandmarkRecognitionTool()
    return tool._recognize_landmark(image_path)