class AppSettings:
    output_folder: str = "../myOutputs/"
    
    """
    Using a commercially available preclinical ultrasound scanner and a scanning acoustic macroscope, 
    the measured speeds of sound were 1547.4 ± 1.4 m/s and 1548.0 ± 6.1 m/s, respectively, 
    and were approximately constant over the frequency range.
    # using CGS units, speed of sound = 154,700 cm/s in water
    """ 
    # Define grid
    GRID_POINTS = 128
    FRAME_SIZE = 30 # cm
    
    GRID_SPACING = FRAME_SIZE/GRID_POINTS
    SIZE = (GRID_POINTS, GRID_POINTS) # Grid size as a tuple (nx, ny)
    
    # Define hardware
    NUM_RECEIVERS_AND_EMITTERS = 32  # Total number of emitters and receivers 
    reference = 32
    SELECTED_EMITTER_INDEX = int(10*NUM_RECEIVERS_AND_EMITTERS/reference)
    SELECTED_RECEIVER_INDEXS1 = int(28*NUM_RECEIVERS_AND_EMITTERS/reference)
    SELECTED_RECEIVER_INDEXS2 = int(20*NUM_RECEIVERS_AND_EMITTERS/reference)
    
    CENTER = (GRID_POINTS*GRID_SPACING/2, GRID_POINTS*GRID_SPACING/2)  # Center of the circular arrangement of emitters and receivers
    RADIUS = GRID_POINTS*GRID_SPACING/2 - GRID_POINTS*GRID_SPACING/2*0.1   # Radius of the circle for placing emitters and receivers
 
    # # Define obstacles as 2D ellipses      
    OBSTACLES = [
        (FRAME_SIZE/3, FRAME_SIZE/2, FRAME_SIZE/6, FRAME_SIZE/10, 1.1),  # Obstacle 1
        (FRAME_SIZE/3*2, FRAME_SIZE/3*2, FRAME_SIZE/10, FRAME_SIZE/6, 1.2),  # Obstacle 2
        (FRAME_SIZE*4/5, FRAME_SIZE/3, FRAME_SIZE/10, FRAME_SIZE/15, 1.3),  # Obstacle 3
    ]
       
    # Define obstacles as 2D ellipses      
    # OBSTACLES = [
    #     (FRAME_SIZE/3, FRAME_SIZE/2, FRAME_SIZE/6, FRAME_SIZE/10, 2),  # Obstacle 1
    #     (FRAME_SIZE/3*2, FRAME_SIZE/3*2, FRAME_SIZE/10, FRAME_SIZE/6, 1.4),  # Obstacle 2
    #     (FRAME_SIZE*4/5, FRAME_SIZE/3, FRAME_SIZE/10, FRAME_SIZE/15, 1.6),  # Obstacle 3
    # ]  
       # (cx, cy, rx, ry, speed), where:
       # cx, cy are the center coordinates of the ellipse,
       # rx, ry are the semi-major and semi-minor axes of the ellipse,
       # and speed is the propagation speed inside the ellipse.
       
    # the above choice means
    # obstacles = [
    #     (10, 15, 5, 3, 2),  # Obstacle 1
    #     (20, 20, 3, 5, 1.5),  # Obstacle 2
    #     (24, 10, 3, 2, 1.2),  # Obstacle 3
 

app_settings = AppSettings()
