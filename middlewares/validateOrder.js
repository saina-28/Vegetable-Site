const pantryModel = require("../models/Pantry")

const validateOrder = async (req, res, next)=>{    
    const cartItems  = req.cartItems;
    try{
        for(let i = 0; i < cartItems.length; i++)
        {
            const pantryItem = await pantryModel.findOne({_id : cartItems[i].itemid})
            if(pantryItem.item_available_qt <= 0 || cartItems[i].quantity <= 0 || pantryItem.item_available_qt < cartItems[i].quantity){
                throw new Error(`Please recheck the Quantity : ${cartItems[i].item_name}, In Stock ${pantryItem.item_available_qt} and In Cart ${cartItems[i].quantity}`)
            }
        }
        next()
        // res.status(200).json(cartItems)
    }
    catch(error)
    {
        // next(error)
        res.status(400).json({error : error.message})
    }
}

module.exports = {validateOrder}