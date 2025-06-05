from typing import Dict, Type, Any, List, Optional, Union
import logging

from .frame import Frame
from .config import config
from .frame_types import FrameImplementation

logger = logging.getLogger(__name__)

class FrameFactory:
    """Factory for creating Frame instances based on configuration."""
    
    _implementation = None
    _frame_class = None
    _implementation_config = None
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize the factory with the implementation from configuration."""
        if cls._frame_class is None:
            impl_name = config.get_frame_implementation()
            cls.set_implementation(impl_name)
    
    @classmethod
    def get_implementation(cls) -> Type[Frame]:
        """Get the current Frame implementation class."""
        if cls._frame_class is None:
            cls.initialize()
        return cls._frame_class
    
    @classmethod
    def set_implementation(cls, implementation: Union[str, FrameImplementation]) -> None:
        """Set the Frame implementation to use.
        
        Args:
            implementation: Implementation name or FrameImplementation enum value
        """
        # Convert enum to string if needed
        if isinstance(implementation, FrameImplementation):
            implementation = implementation.value
            
        # Validate implementation
        if not FrameImplementation.is_valid(implementation):
            error_msg = f"Unknown Frame implementation: {implementation}. Valid options: {FrameImplementation.get_all_implementations()}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        cls._implementation = implementation
        cls._implementation_config = config.get_implementation_config(implementation)
        
        logger.info(f"Setting Frame implementation to '{implementation}'")
        
        if implementation == FrameImplementation.PANDAS.value:
            from .pandas_impl.frame import PandasFrame
            cls._frame_class = PandasFrame
        elif implementation == FrameImplementation.CUDF.value:
            try:
                from .cudf_impl.frame import CudfFrame
                cls._frame_class = CudfFrame
                # Initialize with configuration
                if hasattr(CudfFrame, 'configure'):
                    CudfFrame.configure(cls._implementation_config)
            except ImportError:
                error_msg = f"{FrameImplementation.CUDF.value} implementation requested but cuDF is not installed"
                logger.error(error_msg)
                raise ImportError(error_msg)
        elif implementation == FrameImplementation.MNMG.value:
            try:
                from .mnmg_impl.frame import MNMGFrame
                cls._frame_class = MNMGFrame
                # Initialize with configuration
                if hasattr(MNMGFrame, 'configure'):
                    MNMGFrame.configure(cls._implementation_config)
            except ImportError:
                error_msg = f"{FrameImplementation.MNMG.value} implementation requested but required dependencies are not installed"
                logger.error(error_msg)
                raise ImportError(error_msg)
        elif implementation == FrameImplementation.SCALLOP.value:
            try:
                from .scallop_impl.frame import ScallopFrame
                cls._frame_class = ScallopFrame
                # Initialize with configuration
                if hasattr(ScallopFrame, 'configure'):
                    ScallopFrame.configure(cls._implementation_config)
            except ImportError:
                error_msg = f"{FrameImplementation.SCALLOP.value} implementation requested but Scallop is not installed"
                logger.error(error_msg)
                raise ImportError(error_msg)
    
    @classmethod
    def empty(cls, columns: List[str]) -> Frame:
        """Create an empty frame with given columns."""
        frame_class = cls.get_implementation()
        return frame_class.empty(columns)
    
    @classmethod
    def from_dicts(cls, rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> Frame:
        """Create a frame from a list of dictionaries."""
        frame_class = cls.get_implementation()
        return frame_class.from_dicts(rows, columns)
    
    # Convenience methods to directly use the current Frame implementation

    @classmethod
    def get_make_frame_function(cls):
        """Get a reference to the make_frame function for the current implementation.
        
        This allows database instances to store their own copy of the function,
        ensuring they always use their assigned implementation regardless of
        global FrameFactory changes.
        
        Returns:
            A module with from_dicts and other Frame factory methods
        """
        # Ensure implementation is initialized
        cls.get_implementation()
        
        # Return a reference to the current make_frame module
        # This can be stored by the database to retain its implementation
        from .frame import make_frame
        return make_frame
    
    @classmethod
    def make_frame(cls, data=None, columns=None) -> Frame:
        """Create a Frame instance using the current implementation.
        
        Args:
            data: Input data, can be list of dicts, pandas DataFrame, etc.
            columns: Column names (optional)
            
        Returns:
            A Frame instance of the configured implementation type
        """
        impl_class = cls.get_implementation()
        
        # Handle different input types based on implementation
        try:
            if hasattr(impl_class, "from_data"):
                return impl_class.from_data(data, columns)
            elif data is None and columns is not None:
                return impl_class.empty(columns)
            elif isinstance(data, list) and (not data or isinstance(data[0], dict)):
                return impl_class.from_dicts(data, columns)
            else:
                # Try direct initialization
                return impl_class(data)
        except Exception as e:
            logger.error(f"Error creating {impl_class.__name__}: {e}")
            raise TypeError(f"Cannot create {impl_class.__name__} from data of type {type(data)}")
            
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get the current implementation configuration.
        
        Returns:
            Configuration dictionary for the current implementation
        """
        if cls._implementation_config is None:
            cls.get_implementation()  # Ensures implementation is loaded
        return cls._implementation_config.copy()
        
    @classmethod
    def load_config_from_file(cls, config_file: str) -> None:
        """Load configuration from a YAML file.
        
        Args:
            config_file: Path to YAML configuration file
        """
        # Load configuration
        config.load_from_file(config_file)
        
        # Reset implementation to force reloading with new config
        cls._frame_class = None
        cls._implementation = None
        cls._implementation_config = None
        
    @classmethod
    def get_current_implementation_name(cls) -> Optional[str]:
        """Get the name of the current Frame implementation.
        
        Returns:
            The name of the current implementation or None if not set
        """
        return cls._implementation
    
    @classmethod
    def make_database(cls, implementation: Optional[Union[str, FrameImplementation]] = None):
        """Create a DatalogDatabase instance with the specified Frame implementation.
        
        Args:
            implementation: Optional implementation (FrameImplementation enum or string value)
                           If None, uses the currently set implementation
        
        Returns:
            A DatalogDatabase instance using the specified Frame implementation
        """
        original_impl = cls._implementation
        
        try:
            # If specific implementation requested, temporarily set it
            if implementation:
                cls.set_implementation(implementation)
            else:
                # Otherwise ensure current implementation is initialized
                cls.get_implementation()
                
            # Import here to avoid circular import
            from .database import DatalogDatabase
            return DatalogDatabase()
        finally:
            # Restore original implementation if we changed it
            if implementation and original_impl != implementation:
                if original_impl:
                    cls.set_implementation(original_impl)
                else:
                    cls._implementation = None
                    cls._frame_class = None
                    cls._implementation_config = None
