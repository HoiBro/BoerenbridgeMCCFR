
def identity(hand, hist, pos, mean):
    """No abstraction"""
    return hand, hist


def simple(hand, hist, pos, mean):
    """Abstract away the entire history"""
    return hand, ()


def simple_hand(hand, hist, pos, mean):
    """Abstract away the entire hand"""
    return [], hist


def naive(hand, hist, pos, mean):
    """Least possible states"""
    return [], ()


def advanced(hand, hist, pos, mean):
    """Abstract away the hand to a list containing avg hand, number of suits and number of high cards,"""
    if len(hand) == 0:
        hand_str = 0
    else:
        hand_str = sum(map(lambda x: x[1], hand)) / len(hand)
    
    suits = set()
    for card in hand:
        suits.add(card[0])
    high_cards = len([x for x in hand if x[1] > mean])
    new_hand = [hand_str, len(suits), high_cards]

    return new_hand, hist
