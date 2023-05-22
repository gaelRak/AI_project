import pygame

# Initialize Pygame
pygame.init()

# Set the width and height of the canvas
width, height = 500,100

# Create a new canvas
canvas = pygame.display.set_mode((width, height))
canvas.fill((255, 255, 255))

# Set the drawing color
drawing_color = (0, 0, 0)

# Flag to indicate drawing
drawing = False

# Start the game loop
running = True
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Start drawing
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop drawing
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            # Draw on the canvas
            pygame.draw.circle(canvas, drawing_color, event.pos, 5)

    # Update the canvas
    pygame.display.flip()

# Save the drawn image
pygame.image.save(canvas, "image/created/drawn_image.png")

# Quit Pygame
pygame.quit()
