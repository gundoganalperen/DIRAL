def calculate_ia_penalty(ia):
    """
    This function receives ia array and calculates the sum ia values based on timing slot.
    :param ia:
    :return:
    """
    sum_ia = 0
    for i in range(len(ia)):
        if ia[i] > 0:
            sum_ia += (i+1)*ia[i]

    return sum_ia
