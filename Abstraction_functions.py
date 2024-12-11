
def identity(hand, trump, hist, pos, mean):
    """No abstraction"""
    return hand, trump, hist


def simple(hand, trump, hist, pos, mean):
    """Abstract away the entire history"""
    return hand, trump, ()


def simple_hand(hand, trump, hist, pos, mean):
    """Abstract away the entire hand"""
    return [], trump, hist


def naive(hand, trump, hist, pos, mean):
    """Least possible states"""
    return [], (), ()


def suit(hand, trump, hist, pos, mean):
    """Abstract away the trumps rank"""
    return hand, trump[0], hist


def advanced(hand, trump, hist, pos, mean):
    """Abstract away the hand to a list containing avg hand, number of suits and number of high cards"""
    if len(hand) == 0:
        hand_str = 0
    else:
        hand_str = sum(map(lambda x: x[1], hand)) / len(hand)
    
    suits = set()
    for card in hand:
        suits.add(card[0])
    high_cards = len([x for x in hand if x[1] > mean])
    new_hand = [hand_str, len(suits), high_cards]

    return new_hand, trump, hist
