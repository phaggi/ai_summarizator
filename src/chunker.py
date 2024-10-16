from text_chunker import TextChunker

# Create a new TextChunker object with a maximum chunk length of 50 characters
chunker = TextChunker(maxlen=1000)

# Chunk a long text string into smaller chunks
text = "This is a long text string..."
for chunk in chunker.chunk(text):
    print(chunk)