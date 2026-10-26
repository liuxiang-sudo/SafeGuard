import re
import random
import numpy as np
from .ml_detector import ml_detector

def analyze_user_input(user_data):
    """
    Analyze user input data and determine if the user is a bot or human.
    
    Returns a dictionary with:
    - probability: float between 0 and 1 (1 = definitely human, 0 = definitely bot)
    - evidence: list of findings that contributed to the determination
    """
    # Try to use ML-based approach first
    try:
        bot_probability, ml_evidence = ml_detector.predict_bot_probability(user_data)
        human_probability = 1 - bot_probability
        
        # Determine classification based on probabilities
        if human_probability >= 0.7:
            classification = "Human"
        elif human_probability <= 0.3:
            classification = "Bot"
        else:
            classification = "Uncertain"
            
        return {
            'probability_human': round(human_probability, 2),
            'probability_bot': round(bot_probability, 2),
            'classification': classification,
            'evidence': ml_evidence
        }
    except Exception as e:
        # If ML detection fails, fall back to the rule-based approach
        print(f"ML detection failed: {str(e)}. Falling back to rule-based analysis.")
        return rule_based_analysis(user_data)

def rule_based_analysis(user_data):
    """Legacy rule-based analysis as a fallback method."""
    evidence = []
    scores = []
    
    # Analyze username
    username_score, username_evidence = analyze_username(user_data['username'])
    scores.append(username_score)
    evidence.extend(username_evidence)
    
    # Analyze email
    email_score, email_evidence = analyze_email(user_data['email'])
    scores.append(email_score)
    evidence.extend(email_evidence)
    
    # Analyze bio text
    bio_score, bio_evidence = analyze_bio(user_data['bio'])
    scores.append(bio_score)
    evidence.extend(bio_evidence)
    
    # Analyze time to fill form
    time_score, time_evidence = analyze_time(user_data['time_to_fill'])
    scores.append(time_score)
    evidence.extend(time_evidence)
    
    # Analyze typing pattern
    typing_score, typing_evidence = analyze_typing(user_data['typing_pattern'])
    scores.append(typing_score)
    evidence.extend(typing_evidence)
    
    # Analyze navigation behavior
    nav_score, nav_evidence = analyze_navigation(user_data['navigation'])
    scores.append(nav_score)
    evidence.extend(nav_evidence)
    
    # Calculate final probability
    # Weighted average of all scores
    weights = [0.15, 0.15, 0.20, 0.15, 0.20, 0.15]  # Weights for different factors
    final_probability = np.average(scores, weights=weights)
    
    # Round to 2 decimal places
    final_probability = round(final_probability, 2)
    
    # Determine overall classification
    if final_probability >= 0.7:
        classification = "Human"
    elif final_probability <= 0.3:
        classification = "Bot"
    else:
        classification = "Uncertain"
    
    # Add fallback method note for transparency
    evidence.append("Note: Analysis performed using rule-based approach as fallback method.")
    
    return {
        'probability_human': final_probability,
        'probability_bot': round(1 - final_probability, 2),
        'classification': classification,
        'evidence': evidence
    }

def analyze_username(username):
    evidence = []
    
    if not username:
        evidence.append("No username provided - typical of automated submissions.")
        return 0.2, evidence
    
    # Check for random alphanumeric usernames (common in bots)
    if re.match(r'^[a-zA-Z0-9]{8,}$', username) and any(c.isdigit() for c in username):
        evidence.append(f"Username '{username}' appears randomly generated with numbers and letters.")
        score = 0.3
    # Check for common human username patterns
    elif re.match(r'^[a-zA-Z]+[._]?[a-zA-Z]+$', username):
        evidence.append(f"Username '{username}' follows typical human naming patterns.")
        score = 0.8
    # Check for excessive numbers
    elif sum(c.isdigit() for c in username) > len(username) * 0.5:
        evidence.append(f"Username '{username}' contains an unusually high number of digits.")
        score = 0.4
    else:
        evidence.append(f"Username '{username}' has characteristics of both human and automated users.")
        score = 0.6
        
    return score, evidence

def analyze_email(email):
    evidence = []
    
    if not email:
        evidence.append("No email provided - typical of some automated submissions.")
        return 0.2, evidence
    
    # Check for valid email format
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        evidence.append(f"Email '{email}' has invalid format, suggesting rushed or automated entry.")
        return 0.3, evidence
    
    # Check for disposable email domains
    disposable_domains = ['tempmail', 'throwaway', 'mailinator', 'guerrilla', 'temp-mail', 'temp.com']
    if any(domain in email.lower() for domain in disposable_domains):
        evidence.append(f"Email '{email}' uses a disposable email service, common in automated signups.")
        score = 0.2
    # Check for major email providers (more likely human)
    elif any(domain in email.lower() for domain in ['gmail', 'yahoo', 'hotmail', 'outlook', 'icloud']):
        evidence.append(f"Email '{email}' uses a common provider, typical of regular users.")
        score = 0.8
    # Random-looking email address
    elif re.match(r'^[a-z0-9]{8,}@', email.lower()):
        evidence.append(f"Email '{email}' appears to have a randomly generated local part.")
        score = 0.4
    else:
        evidence.append(f"Email '{email}' has usual characteristics.")
        score = 0.6
        
    return score, evidence

def analyze_bio(bio):
    evidence = []
    
    if not bio:
        evidence.append("No bio text provided.")
        return 0.5, evidence
    
    # Check for very short bios
    if len(bio) < 10:
        evidence.append("Bio is unusually short, which can indicate lack of human input.")
        return 0.4, evidence
    
    # Check for repeated characters
    if any(bio.count(char) > len(bio) * 0.3 for char in set(bio)):
        evidence.append("Bio contains unusually repetitive characters.")
        score = 0.3
    # Check for coherent sentences
    elif re.search(r'[.!?]\s+[A-Z]', bio) or len(bio.split()) > 5:
        evidence.append("Bio contains proper sentence structure, typical of human writing.")
        score = 0.8
    # Check for lack of spaces (bot-like)
    elif bio.count(' ') < len(bio) * 0.1:
        evidence.append("Bio lacks appropriate spacing between words.")
        score = 0.3
    else:
        evidence.append("Bio text has mixed characteristics.")
        score = 0.6
        
    return score, evidence

def analyze_time(time_to_fill):
    evidence = []
    
    try:
        time_seconds = float(time_to_fill)
    except:
        evidence.append("Unable to parse form completion time.")
        return 0.5, evidence
    
    if time_seconds < 5:
        evidence.append(f"Form completed extremely quickly ({time_seconds}s), strongly suggesting automated submission.")
        score = 0.1
    elif time_seconds < 15:
        evidence.append(f"Form completed very quickly ({time_seconds}s), potential automated behavior.")
        score = 0.3
    elif 15 <= time_seconds <= 180:
        evidence.append(f"Form completion time ({time_seconds}s) is within normal human range.")
        score = 0.9
    else:
        evidence.append(f"Form completion time ({time_seconds}s) is unusually long.")
        score = 0.7
        
    return score, evidence

def analyze_typing(typing_pattern):
    evidence = []
    
    if not typing_pattern:
        evidence.append("No typing pattern data available.")
        return 0.5, evidence
    
    try:
        # Assuming typing_pattern is a comma-separated list of keystroke intervals
        intervals = [float(i) for i in typing_pattern.split(',')]
        
        # Analyze variation in typing speed (humans have more natural variation)
        std_dev = np.std(intervals) if len(intervals) > 1 else 0
        
        if std_dev < 50:
            evidence.append("Typing pattern shows unusually uniform timing, suggesting automated input.")
            score = 0.2
        elif 50 <= std_dev <= 500:
            evidence.append("Typing pattern shows natural human variation in rhythm.")
            score = 0.9
        else:
            evidence.append("Typing pattern shows extreme variation, possibly indicating user distraction.")
            score = 0.7
    except:
        evidence.append("Typing pattern data provided but couldn't be analyzed.")
        score = 0.5
        
    return score, evidence

def analyze_navigation(navigation):
    evidence = []
    
    if not navigation:
        evidence.append("No navigation pattern data available.")
        return 0.5, evidence
    
    # Simple analysis based on navigation count
    try:
        nav_count = int(navigation)
        
        if nav_count == 0:
            evidence.append("No previous page navigation, direct form access could indicate automated submission.")
            score = 0.3
        elif 1 <= nav_count <= 5:
            evidence.append(f"User navigated through {nav_count} pages, typical of normal browsing behavior.")
            score = 0.8
        else:
            evidence.append(f"Unusually high navigation count ({nav_count}), could indicate unusual behavior.")
            score = 0.6
    except:
        # If it's not just a count but a detailed pattern
        if 'direct' in navigation.lower():
            evidence.append("Direct access pattern, potential bot behavior.")
            score = 0.4
        elif 'browsed' in navigation.lower():
            evidence.append("Normal browsing pattern detected, typical of human users.")
            score = 0.8
        else:
            evidence.append("Navigation pattern inconclusive.")
            score = 0.5
            
    return score, evidence 