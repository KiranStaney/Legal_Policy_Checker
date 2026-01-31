try:
    from google import genai
    print("SUCCESS: 'google.genai' imported correctly!")
except ImportError as e:
    print(f"FAILURE: {e}")
    
    # Check what IS inside google
    import google
    print(f"Google package location: {google.__path__}")
    try:
        import google.generativeai
        print("Note: 'google.generativeai' (older SDK) IS installed.")
    except ImportError:
        pass