@app.route('/ingredients')
@login_required
def ingredients():
    # Your existing code to get frequent_itemsets
    # ...
    
    # Prepare network data for D3.js visualization
    network_data = {"nodes": [], "links": []}
    
    # Create a set of all ingredients
    ingredient_set = set()
    for item in frequent_itemsets:
        for ingredient in item.itemsets:
            ingredient_set.add(ingredient)
    
    # Add nodes
    for ingredient in ingredient_set:
        network_data["nodes"].append({
            "id": ingredient,
            "group": 1,
            "size": 1
        })
    
    # Add links
    for item in frequent_itemsets:
        if len(item.itemsets) > 1:
            for i in range(len(item.itemsets)):
                for j in range(i + 1, len(item.itemsets)):
                    network_data["links"].append({
                        "source": item.itemsets[i],
                        "target": item.itemsets[j],
                        "value": item.support,
                        "strength": item.support
                    })
    
    return render_template('ingredients.html', 
                          frequent_itemsets=frequent_itemsets,
                          network_data=network_data)