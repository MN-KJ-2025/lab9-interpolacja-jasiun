# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(n, int) and n > 0):
        return None
    
    k = np.arange(n)
    nodes = np.cos(k * np.pi / (n - 1))
    
    return nodes

def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(n, int) and n > 0):
        return None

    weights = np.ones(n)

    weights[1::2] = -1

    weights[0] *= 0.5
    weights[-1] *= 0.5

    return weights


def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(xi, np.ndarray) and isinstance(yi, np.ndarray) and isinstance(wi, np.ndarray) and isinstance(x, np.ndarray)):
        return None
    
    if not (len(xi) == len(yi) and len(yi) == len(wi)):
        return None
    
    results = []
    for x_val in x:
        match = np.where(np.isclose(xi, x_val))[0]
        if len(match) > 0:
            results.append(yi[match[0]])
        else:
            L = wi / (x_val - xi)
            results.append(np.dot(yi, L) / np.sum(L))
    
    return np.array(results)


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(xr, (int, float, list, np.ndarray)) and isinstance(x,(int, float, list, np.ndarray))):
        return None

    x1 = np.asarray(xr)
    x2 = np.asarray(x)

    if(x1.shape != x2.shape):
        return None
    
    return np.max(abs(x1 - x2))